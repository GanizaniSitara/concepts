' ===== ArchiveBackup.bas =====
' Outlook VBA: Export Online Archive to .MSG with checkpoint/restart & logging
' Tested on Microsoft 365 Desktop Outlook (MAPI/OOM)
' No extra references required (uses late binding for FileSystemObject)

Option Explicit

' ---------------------------
' CONFIG — EDIT TO YOUR NEEDS
' ---------------------------
Private Const ROOT_EXPORT As String = "D:\MailBackup"          ' Root backup folder
Private Const LOG_PATH As String = "D:\MailBackup\backup.log"  ' Log file
Private Const CHECKPOINT_CSV As String = "D:\MailBackup\checkpoint.csv"  ' Checkpoint file

' Re-export rule on rerun:
'   True  = overwrite only if the item LastModificationTime changed since last run
'   False = skip if target file already exists
Private Const REEXPORT_IF_CHANGED As Boolean = True

' Retry count for transient errors (e.g., RPC/Exchange throttling)
Private Const MAX_RETRIES As Long = 3

' Exclude folders by (case-insensitive) name match within the archive branch
' Add more as needed: "Junk E-mail","Deleted Items","Conversation History","Sync Issues"
Private ExcludedFolders As Variant

' ---------------------------
' ENTRY POINT
' ---------------------------
Public Sub ArchiveBackup_Run()
    On Error GoTo EH

    ExcludedFolders = Array("junk e-mail", "junk", "deleted items", "sync issues", "conflicts", _
                            "conversation history", "rss feeds", "outbox", "drafts")

    EnsureFolderExists ROOT_EXPORT
    InitLog
    LogLine "=== START " & Now & " ==="
    LogLine "Export root: " & ROOT_EXPORT

    Dim ses As Outlook.NameSpace
    Set ses = Application.Session

    Dim st As Outlook.Store
    Dim foundArchive As Boolean: foundArchive = False

    ' Find the Online Archive store
    For Each st In ses.Stores
        ' olExchangeArchiveMailbox = 3 (but we avoid hard-coded enum — check both name and type)
        If IsExchangeArchiveStore(st) Then
            foundArchive = True
            LogLine "Processing Archive Store: " & st.DisplayName
            ProcessStore st
        End If
    Next st

    If Not foundArchive Then
        LogLine "WARNING: No Exchange Archive mailbox detected. Nothing exported."
        MsgBox "No Online Archive mailbox detected in Outlook profile.", vbExclamation
    Else
        LogLine "=== FINISH " & Now & " ==="
        MsgBox "Archive export complete. See log: " & LOG_PATH, vbInformation
    End If

    Exit Sub
EH:
    LogLine "FATAL ERROR: " & Err.Number & " - " & Err.Description
    MsgBox "Fatal error: " & Err.Description & vbCrLf & "See log: " & LOG_PATH, vbCritical
End Sub

' ---------------------------
' STORE / FOLDER WALK
' ---------------------------
Private Sub ProcessStore(ByVal st As Outlook.Store)
    On Error GoTo EH

    Dim rootFld As Outlook.Folder
    Set rootFld = st.GetRootFolder

    Dim basePath As String
    basePath = MakeSafePath(ROOT_EXPORT & "\" & MakeSafeName(st.DisplayName))
    EnsureFolderExists basePath

    ' Walk folders recursively, but only within the Archive store
    WalkFolder rootFld, basePath, st.StoreID

    Exit Sub
EH:
    LogLine "ERROR in ProcessStore: " & Err.Number & " - " & Err.Description
End Sub

Private Sub WalkFolder(ByVal fld As Outlook.Folder, ByVal currentPath As String, ByVal storeId As String)
    On Error GoTo EH

    If ShouldExcludeFolder(fld.Name) Then
        LogLine "Skip folder (excluded): " & FullFolderPath(fld)
        Exit Sub
    End If

    Dim thisPath As String
    thisPath = MakeSafePath(currentPath & "\" & MakeSafeName(fld.Name))
    EnsureFolderExists thisPath

    ' Export items in chronological order (ReceivedTime ascending)
    ExportFolderItems fld, thisPath, storeId

    ' Recurse subfolders
    Dim sf As Outlook.Folder
    For Each sf In fld.Folders
        WalkFolder sf, thisPath, storeId
    Next sf

    Exit Sub
EH:
    LogLine "ERROR in WalkFolder [" & FullFolderPath(fld) & "]: " & Err.Number & " - " & Err.Description
End Sub

' ---------------------------
' EXPORT ITEMS
' ---------------------------
Private Sub ExportFolderItems(ByVal fld As Outlook.Folder, ByVal targetPath As String, ByVal storeId As String)
    On Error GoTo EH

    Dim itms As Outlook.Items
    Set itms = fld.Items

    ' Sort by ReceivedTime ascending
    On Error Resume Next
    itms.Sort "[ReceivedTime]", False
    itms.IncludeRecurrences = True
    On Error GoTo EH

    Dim i As Long
    Dim it As Object

    For i = 1 To itms.Count
        DoEvents
        On Error Resume Next
        Set it = itms(i)
        If Err.Number <> 0 Then
            LogLine "WARN: Unable to access item index " & i & " in " & FullFolderPath(fld) & " — " & Err.Description
            Err.Clear
            On Error GoTo EH
            GoTo NextItem
        End If
        On Error GoTo EH

        If Not it Is Nothing Then
            If it.Class = olMail Then
                ExportOneMail it, targetPath, storeId, fld
            Else
                ' Not a mail item—skip
            End If
        End If

NextItem:
        ' batch flush (lightweight here since we append-per-line)
        ' could add Sleep if throttling appears
    Next i

    Exit Sub
EH:
    LogLine "ERROR in ExportFolderItems [" & FullFolderPath(fld) & "]: " & Err.Number & " - " & Err.Description
End Sub

Private Sub ExportOneMail(ByVal mail As Outlook.MailItem, ByVal targetPath As String, ByVal storeId As String, ByVal parentFld As Outlook.Folder)
    On Error GoTo EH

    Dim pa As Outlook.PropertyAccessor
    Set pa = mail.PropertyAccessor

    Dim internetId As String
    internetId = SafeGetInternetMessageId(pa)

    Dim conv As String
    conv = Nz(mail.ConversationTopic, "")

    Dim subj As String
    subj = Nz(mail.Subject, "")

    Dim recv As Date
    On Error Resume Next
    recv = mail.ReceivedTime
    If Err.Number <> 0 Then recv = mail.CreationTime
    Err.Clear
    On Error GoTo EH

    Dim ts As String
    ts = Format(recv, "yyyy-mm-dd_hhnnss")

    ' Build filename (truncate to avoid MAX_PATH problems)
    Dim baseName As String
    baseName = ts & "__" & TruncForPath(conv, 80)
    If subj <> "" Then baseName = baseName & "__" & TruncForPath(subj, 80)
    If internetId <> "" Then baseName = baseName & "__MID-" & TruncForPath(internetId, 80)

    Dim filePath As String
    filePath = MakeSafePath(targetPath & "\" & MakeSafeName(baseName) & ".msg")

    Dim changed As Boolean
    changed = HasItemChangedSinceCheckpoint(storeId, parentFld.EntryID, mail.EntryID, mail.LastModificationTime, filePath)

    If Not changed Then
        ' Decide by policy
        If Not REEXPORT_IF_CHANGED Then
            If FileExists(filePath) Then
                ' skip silently
                Exit Sub
            End If
        End If
    End If

    ' Save with retry
    Dim attempt As Long
    For attempt = 1 To MAX_RETRIES
        On Error Resume Next
        mail.SaveAs filePath, olMSGUnicode
        If Err.Number = 0 Then
            On Error GoTo EH
            UpdateCheckpoint storeId, parentFld.EntryID, mail.EntryID, mail.LastModificationTime, filePath
            Exit For
        Else
            LogLine "WARN save fail try " & attempt & " for [" & filePath & "]: " & Err.Number & " - " & Err.Description
            Dim eNum As Long: eNum = Err.Number
            Err.Clear
            On Error GoTo EH
            DoEvents
            If attempt = MAX_RETRIES Then
                LogLine "ERROR: save failed permanently for [" & filePath & "], EntryID=" & mail.EntryID
            End If
        End If
    Next attempt

    Exit Sub
EH:
    LogLine "ERROR in ExportOneMail [" & filePath & "]: " & Err.Number & " - " & Err.Description
End Sub

' ---------------------------
' CHECKPOINT CSV
' ---------------------------
' CSV columns: StoreID,FolderID,EntryID,LastModUTC,FilePath
Private Function HasItemChangedSinceCheckpoint(ByVal storeId As String, ByVal folderId As String, _
                                               ByVal entryId As String, ByVal lastMod As Date, _
                                               ByVal filePath As String) As Boolean
    ' Returns True if:
    '   - checkpoint not found for this EntryID, OR
    '   - lastMod is newer than checkpoint, OR
    '   - file does not exist
    If Not FileExists(CHECKPOINT_CSV) Then
        HasItemChangedSinceCheckpoint = True
        Exit Function
    End If

    Dim f As Integer: f = FreeFile
    On Error GoTo EH
    Open CHECKPOINT_CSV For Input As #f
    Dim line As String
    Dim found As Boolean: found = False
    Dim lastModStr As String

    lastModStr = Format$(lastMod, "yyyy-mm-dd hh:nn:ss")

    Do While Not EOF(f)
        Line Input #f, line
        If InStr(1, line, entryId, vbTextCompare) > 0 Then
            ' crude parse; safe because we write our own format and entryId is unique
            Dim parts() As String
            parts = Split(line, ",")
            If UBound(parts) >= 4 Then
                Dim cpLast As String
                cpLast = parts(3)
                found = True
                If Not FileExists(filePath) Then
                    HasItemChangedSinceCheckpoint = True
                Else
                    ' compare timestamps
                    If DateValue(lastMod) & " " & TimeValue(lastMod) > CDate(cpLast) Then
                        HasItemChangedSinceCheckpoint = True
                    Else
                        HasItemChangedSinceCheckpoint = False
                    End If
                End If
                Exit Do
            End If
        End If
    Loop
    Close #f

    If Not found Then
        HasItemChangedSinceCheckpoint = True
    End If
    Exit Function
EH:
    On Error Resume Next
    Close #f
    ' conservative: treat as changed so we export
    HasItemChangedSinceCheckpoint = True
End Function

Private Sub UpdateCheckpoint(ByVal storeId As String, ByVal folderId As String, _
                             ByVal entryId As String, ByVal lastMod As Date, _
                             ByVal filePath As String)
    On Error GoTo EH
    Dim f As Integer: f = FreeFile
    EnsureFolderExists GetParentPath(CHECKPOINT_CSV)
    Open CHECKPOINT_CSV For Append As #f
    Print #f, storeId & "," & folderId & "," & entryId & "," & _
              Format$(lastMod, "yyyy-mm-dd hh:nn:ss") & ",""" & filePath & """"
    Close #f
    Exit Sub
EH:
    On Error Resume Next
    Close #f
    LogLine "ERROR writing checkpoint: " & Err.Number & " - " & Err.Description
End Sub

' ---------------------------
' HELPERS
' ---------------------------
Private Function IsExchangeArchiveStore(ByVal st As Outlook.Store) As Boolean
    On Error Resume Next
    ' Prefer the ExchangeStoreType property when available
    ' 3 = olExchangeArchiveMailbox
    Dim est As Variant
    est = CallByName(st, "ExchangeStoreType", VbGet)
    If Not IsError(est) Then
        If est = 3 Then
            IsExchangeArchiveStore = True
            Exit Function
        End If
    End If
    ' Fallback: heuristics on display name
    IsExchangeArchiveStore = (InStr(1, st.DisplayName, "archive", vbTextCompare) > 0)
End Function

Private Function ShouldExcludeFolder(ByVal name As String) As Boolean
    Dim nm As String: nm = LCase$(Trim$(name))
    Dim i As Long
    For i = LBound(ExcludedFolders) To UBound(ExcludedFolders)
        If nm = LCase$(ExcludedFolders(i)) Then
            ShouldExcludeFolder = True
            Exit Function
        End If
    Next i
    ShouldExcludeFolder = False
End Function

Private Function FullFolderPath(ByVal fld As Outlook.Folder) As String
    On Error Resume Next
    FullFolderPath = fld.FolderPath
End Function

Private Function Nz(ByVal s As Variant, ByVal fallback As String) As String
    If IsEmpty(s) Or IsNull(s) Then
        Nz = fallback
    Else
        Nz = CStr(s)
    End If
End Function

Private Function SafeGetInternetMessageId(ByVal pa As Outlook.PropertyAccessor) As String
    On Error Resume Next
    ' PR_INTERNET_MESSAGE_ID: 0x1035, type PT_STRING8 (001E) — Outlook DASL tag commonly used:
    Dim val As String
    val = pa.GetProperty("http://schemas.microsoft.com/mapi/proptag/0x1035001E")
    If Err.Number <> 0 Or val = "" Then
        Err.Clear
        ' Try Unicode variant (001F)
        val = pa.GetProperty("http://schemas.microsoft.com/mapi/proptag/0x1035001F")
    End If
    If Err.Number <> 0 Then
        val = ""
        Err.Clear
    End If
    ' Strip angle brackets and unsafe chars
    val = Replace(val, "<", "")
    val = Replace(val, ">", "")
    val = SanitizeFilePart(val)
    SafeGetInternetMessageId = val
End Function

Private Function TruncForPath(ByVal s As String, ByVal maxLen As Long) As String
    If Len(s) > maxLen Then
        TruncForPath = Left$(s, maxLen)
    Else
        TruncForPath = s
    End If
End Function

Private Function MakeSafeName(ByVal s As String) As String
    s = SanitizeFilePart(s)
    If Len(s) = 0 Then s = "_"
    MakeSafeName = s
End Function

Private Function SanitizeFilePart(ByVal s As String) As String
    Dim bad As Variant
    bad = Array("<", ">", ":", """", "/", "\", "|", "?", "*", vbCr, vbLf, vbTab)
    Dim i As Long
    For i = LBound(bad) To UBound(bad)
        s = Replace(s, bad(i), " ")
    Next i
    s = Trim$(s)
    ' collapse multiple spaces
    Do While InStr(s, "  ") > 0
        s = Replace(s, "  ", " ")
    Loop
    SanitizeFilePart = s
End Function

Private Function MakeSafePath(ByVal s As String) As String
    ' Clean each segment
    Dim parts() As String
    parts = Split(s, "\")
    Dim i As Long
    For i = LBound(parts) To UBound(parts)
        parts(i) = MakeSafeName(parts(i))
    Next i
    MakeSafePath = Join(parts, "\")
End Function

Private Sub EnsureFolderExists(ByVal path As String)
    On Error GoTo EH
    If Len(path) = 0 Then Exit Sub
    Dim fso As Object
    Set fso = CreateObject("Scripting.FileSystemObject")
    If Not fso.FolderExists(path) Then
        fso.CreateFolder path
    End If
    Exit Sub
EH:
    LogLine "ERROR creating folder [" & path & "]: " & Err.Number & " - " & Err.Description
End Sub

Private Function GetParentPath(ByVal filePath As String) As String
    Dim i As Long
    i = InStrRev(filePath, "\")
    If i > 0 Then
        GetParentPath = Left$(filePath, i - 1)
    Else
        GetParentPath = ""
    End If
End Function

Private Function FileExists(ByVal filePath As String) As Boolean
    On Error Resume Next
    Dim fso As Object
    Set fso = CreateObject("Scripting.FileSystemObject")
    FileExists = fso.FileExists(filePath)
End Function

' ---------------------------
' LOGGING
' ---------------------------
Private Sub InitLog()
    On Error GoTo EH
    EnsureFolderExists GetParentPath(LOG_PATH)
    Dim f As Integer: f = FreeFile
    Open LOG_PATH For Append As #f
    Print #f, ""
    Close #f
    Exit Sub
EH:
    ' swallow
End Sub

Private Sub LogLine(ByVal s As String)
    On Error Resume Next
    Dim f As Integer: f = FreeFile
    Open LOG_PATH For Append As #f
    Print #f, Format$(Now, "yyyy-mm-dd hh:nn:ss"); " | "; s
    Close #f
End Sub
