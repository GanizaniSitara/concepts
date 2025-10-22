' ===== ArchiveBackup.bas =====
' Outlook VBA: Export Online Archive to .MSG with checkpoint/restart & logging
' Includes: safe filename/path builder (avoids STG_E_INVALIDNAME) and all helpers.

Option Explicit

' ---------------------------
' CONFIG — EDIT THESE
' ---------------------------
Private Const ROOT_EXPORT As String = "D:\MailBackup"                   ' Root backup folder
Private Const LOG_PATH As String = "D:\MailBackup\backup.log"           ' Log file
Private Const CHECKPOINT_CSV As String = "D:\MailBackup\checkpoint.csv" ' Checkpoint CSV

' Re-export rule on rerun:
'   True  = overwrite only if the item LastModificationTime changed since last run
'   False = skip if target file already exists
Private Const REEXPORT_IF_CHANGED As Boolean = True

' Retry count for transient errors (e.g., RPC/Exchange throttling)
Private Const MAX_RETRIES As Long = 3

' Exclude folders by (case-insensitive) name
Private ExcludedFolders As Variant

' ---------------------------
' PATH/LENGTH SAFEGUARDS
' ---------------------------
Private Const MAX_FULLPATH As Long = 240   ' keep under ~260 for COM/OOM saves
Private Const MAX_FILENAME As Long = 120   ' cap filename segment

' ===========================
' ENTRY POINT
' ===========================
Public Sub ArchiveBackup_Run()
    Dim ses As Outlook.NameSpace
    Dim st As Outlook.Store
    Dim foundArchive As Boolean
    foundArchive = False

    On Error GoTo EH

    ExcludedFolders = Array("junk e-mail", "junk", "deleted items", "sync issues", "conflicts", _
                            "conversation history", "rss feeds", "outbox", "drafts")

    EnsureFolderExists ROOT_EXPORT
    InitLog
    LogLine "=== START " & Now & " ==="
    LogLine "Export root: " & ROOT_EXPORT

    Set ses = Application.Session

    For Each st In ses.Stores
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

' ===========================
' STORE / FOLDER WALK
' ===========================
Private Sub ProcessStore(ByVal st As Outlook.Store)
    Dim rootFld As Outlook.Folder
    Dim basePath As String

    On Error GoTo EH

    Set rootFld = st.GetRootFolder
    basePath = MakeSafePath(ROOT_EXPORT & "\" & MakeSafeName(st.DisplayName))
    EnsureFolderExists basePath

    WalkFolder rootFld, basePath, st.StoreID
    Exit Sub
EH:
    LogLine "ERROR in ProcessStore: " & Err.Number & " - " & Err.Description
End Sub

Private Sub WalkFolder(ByVal fld As Outlook.Folder, ByVal currentPath As String, ByVal storeId As String)
    Dim thisPath As String
    Dim sf As Outlook.Folder

    On Error GoTo EH

    If ShouldExcludeFolder(fld.Name) Then
        LogLine "Skip folder (excluded): " & FullFolderPath(fld)
        Exit Sub
    End If

    thisPath = MakeSafePath(currentPath & "\" & MakeSafeName(fld.Name))
    EnsureFolderExists thisPath

    ExportFolderItems fld, thisPath, storeId

    For Each sf In fld.Folders
        WalkFolder sf, thisPath, storeId
    Next sf
    Exit Sub
EH:
    LogLine "ERROR in WalkFolder [" & FullFolderPath(fld) & "]: " & Err.Number & " - " & Err.Description
End Sub

' ===========================
' EXPORT ITEMS
' ===========================
Private Sub ExportFolderItems(ByVal fld As Outlook.Folder, ByVal targetPath As String, ByVal storeId As String)
    Dim itms As Outlook.Items
    Dim i As Long
    Dim it As Object

    On Error GoTo EH

    Set itms = fld.Items
    On Error Resume Next
    itms.Sort "[ReceivedTime]", False
    On Error GoTo EH

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
            End If
        End If
NextItem:
    Next i
    Exit Sub
EH:
    LogLine "ERROR in ExportFolderItems [" & FullFolderPath(fld) & "]: " & Err.Number & " - " & Err.Description
End Sub

Private Sub ExportOneMail(ByVal mail As Outlook.MailItem, ByVal targetPath As String, ByVal storeId As String, ByVal parentFld As Outlook.Folder)
    Dim eNum As Long
    Dim pa As Outlook.PropertyAccessor
    Dim internetId As String
    Dim conv As String
    Dim subj As String
    Dim recv As Date
    Dim ts As String
    Dim baseName As String
    Dim filePath As String
    Dim changed As Boolean
    Dim attempt As Long

    On Error GoTo EH

    Set pa = mail.PropertyAccessor
    internetId = SafeGetInternetMessageId(pa)
    conv = Nz(mail.ConversationTopic, "")
    subj = Nz(mail.Subject, "")

    On Error Resume Next
    recv = mail.ReceivedTime
    If Err.Number <> 0 Then recv = mail.CreationTime
    Err.Clear
    On Error GoTo EH

    ts = Format(recv, "yyyy-mm-dd_hhnnss")

    baseName = ts & "__" & TruncForPath(conv, 80)
    If subj <> "" Then baseName = baseName & "__" & TruncForPath(subj, 80)
    If internetId <> "" Then baseName = baseName & "__MID-" & TruncForPath(internetId, 80)

    filePath = BuildSafeFilePath(targetPath, baseName, mail.EntryID)

    changed = HasItemChangedSinceCheckpoint(storeId, parentFld.EntryID, mail.EntryID, mail.LastModificationTime, filePath)

    If Not changed Then
        If Not REEXPORT_IF_CHANGED Then
            If FileExists(filePath) Then Exit Sub
        End If
    End If

    For attempt = 1 To MAX_RETRIES
        On Error Resume Next
        mail.SaveAs filePath, olMSGUnicode
        eNum = Err.Number
        If eNum = 0 Then
            On Error GoTo EH
            UpdateCheckpoint storeId, parentFld.EntryID, mail.EntryID, mail.LastModificationTime, filePath
            Exit For
        Else
            LogLine "WARN save fail try " & attempt & " for [" & filePath & "]: " & eNum & " - " & Err.Description
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

' ===========================
' CHECKPOINT CSV
' ===========================
' CSV columns: StoreID,FolderID,EntryID,LastModUTC,FilePath
Private Function HasItemChangedSinceCheckpoint(ByVal storeId As String, ByVal folderId As String, _
                                               ByVal entryId As String, ByVal lastMod As Date, _
                                               ByVal filePath As String) As Boolean
    Dim f As Integer
    Dim line As String
    Dim found As Boolean
    Dim parts() As String
    Dim cpLast As String

    If Not FileExists(CHECKPOINT_CSV) Then
        HasItemChangedSinceCheckpoint = True
        Exit Function
    End If

    f = FreeFile
    On Error GoTo EH
    Open CHECKPOINT_CSV For Input As #f

    found = False
    Do While Not EOF(f)
        Line Input #f, line
        If InStr(1, line, entryId, vbTextCompare) > 0 Then
            parts = Split(line, ",")
            If UBound(parts) >= 4 Then
                cpLast = parts(3)
                found = True
                If Not FileExists(filePath) Then
                    HasItemChangedSinceCheckpoint = True
                Else
                    ' Compare as doubles (safer than string compare)
                    If CDbl(lastMod) > CDbl(CDate(cpLast)) Then
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

    If Not found Then HasItemChangedSinceCheckpoint = True
    Exit Function
EH:
    On Error Resume Next
    Close #f
    HasItemChangedSinceCheckpoint = True
End Function

Private Sub UpdateCheckpoint(ByVal storeId As String, ByVal folderId As String, _
                             ByVal entryId As String, ByVal lastMod As Date, _
                             ByVal filePath As String)
    Dim f As Integer

    On Error GoTo EH
    f = FreeFile
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

' ===========================
' HELPERS
' ===========================
Private Function IsExchangeArchiveStore(ByVal st As Outlook.Store) As Boolean
    Dim est As Variant
    On Error Resume Next
    est = CallByName(st, "ExchangeStoreType", VbGet)
    If Not IsError(est) Then
        If est = 3 Then
            IsExchangeArchiveStore = True
            Exit Function
        End If
    End If
    IsExchangeArchiveStore = (InStr(1, st.DisplayName, "archive", vbTextCompare) > 0)
End Function

Private Function ShouldExcludeFolder(ByVal name As String) As Boolean
    Dim nm As String
    Dim i As Long
    nm = LCase$(Trim$(name))
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
    Dim val As String
    On Error Resume Next
    val = pa.GetProperty("http://schemas.microsoft.com/mapi/proptag/0x1035001E")
    If Err.Number <> 0 Or val = "" Then
        Err.Clear
        val = pa.GetProperty("http://schemas.microsoft.com/mapi/proptag/0x1035001F")
    End If
    If Err.Number <> 0 Then
        val = ""
        Err.Clear
    End If
    val = Replace$(val, "<", "")
    val = Replace$(val, ">", "")
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

' Tightened sanitization
Private Function SanitizeFilePart(ByVal s As String) As String
    Dim i As Long, ch As String, outS As String
    If Len(s) = 0 Then SanitizeFilePart = "": Exit Function

    Dim bad As Variant
    bad = Array("<", ">", ":", """", "/", "\", "|", "?", "*")
    For i = LBound(bad) To UBound(bad)
        s = Replace$(s, bad(i), " ")
    Next i

    outS = ""
    For i = 1 To Len(s)
        ch = Mid$(s, i, 1)
        If AscW(ch) >= 32 Then outS = outS & ch
    Next i

    Do While InStr(outS, "  ") > 0
        outS = Replace$(outS, "  ", " ")
    Loop
    Do While InStr(outS, "..") > 0
        outS = Replace$(outS, "..", ".")
    Loop

    outS = Trim$(outS)
    Do While Len(outS) > 0 And (Right$(outS, 1) = "." Or Right$(outS, 1) = " ")
        outS = Left$(outS, Len(outS) - 1)
    Loop

    SanitizeFilePart = outS
End Function

Private Function MakeSafePath(ByVal s As String) As String
    Dim parts() As String
    Dim i As Long
    parts = Split(s, "\")
    For i = LBound(parts) To UBound(parts)
        parts(i) = MakeSafeName(parts(i))
    Next i
    MakeSafePath = Join(parts, "\")
End Function

Private Sub EnsureFolderExists(ByVal path As String)
    Dim fso As Object
    On Error GoTo EH
    If Len(path) = 0 Then Exit Sub
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
    Dim fso As Object
    On Error Resume Next
    Set fso = CreateObject("Scripting.FileSystemObject")
    FileExists = fso.FileExists(filePath)
End Function

' ---------------------------
' Minimal targeted fix helpers
' ---------------------------
Private Function IsReservedWinName(ByVal s As String) As Boolean
    Dim n As String
    Dim dotPos As Long
    n = UCase$(Trim$(s))
    dotPos = InStrRev(n, ".")
    If dotPos > 0 Then n = Left$(n, dotPos - 1)
    Select Case n
        Case "CON", "PRN", "AUX", "NUL", _
             "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", _
             "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
            IsReservedWinName = True
        Case Else
            IsReservedWinName = False
    End Select
End Function

Private Function SimpleHash8(ByVal s As String) As String
    Dim i As Long, h As Long
    h = &H1505&
    For i = 1 To Len(s)
        h = ((h * 33) Xor AscW(Mid$(s, i, 1))) And &H7FFFFFFF
    Next i
    SimpleHash8 = Right$("00000000" & Hex$(h), 8)
End Function

Private Function BuildSafeFilePath( _
        ByVal folderPath As String, _
        ByVal baseName As String, _
        ByVal entryId As String) As String

    Dim namePart As String
    Dim fullPath As String
    Dim ts As String, pos As Long, rest As String
    Dim shortCore As String
    Dim newName As String

    namePart = MakeSafeName(baseName)

    If IsReservedWinName(namePart) Then namePart = namePart & "_"
    If Len(namePart) > MAX_FILENAME Then
        namePart = Left$(namePart, MAX_FILENAME)
        Do While Len(namePart) > 0 And (Right$(namePart, 1) = "." Or Right$(namePart, 1) = " ")
            namePart = Left$(namePart, Len(namePart) - 1)
        Loop
    End If

    fullPath = MakeSafePath(folderPath) & "\" & namePart & ".msg"
    If Len(fullPath) <= MAX_FULLPATH Then
        BuildSafeFilePath = fullPath
        Exit Function
    End If

    ' Too long: preserve timestamp prefix and replace rest with EntryID hash
    pos = InStr(1, namePart, "__")
    If pos > 0 Then
        ts = Left$(namePart, pos - 1)
        rest = Mid$(namePart, pos + 2)
    Else
        ts = ""
        rest = namePart
    End If

    shortCore = "MID-" & SimpleHash8(entryId)

    If ts <> "" Then
        newName = ts & "__" & shortCore
    Else
        newName = shortCore
    End If

    fullPath = MakeSafePath(folderPath) & "\" & newName & ".msg"

    If Len(fullPath) > MAX_FULLPATH Then
        newName = Left$(newName, MAX_FILENAME \ 2) & "_" & SimpleHash8(entryId)
        fullPath = MakeSafePath(folderPath) & "\" & newName & ".msg"
    End If

    If IsReservedWinName(newName) Then newName = newName & "_"
    Do While Len(newName) > 0 And (Right$(newName, 1) = "." Or Right$(newName, 1) = " ")
        newName = Left$(newName, Len(newName) - 1)
    Loop

    BuildSafeFilePath = MakeSafePath(folderPath) & "\" & newName & ".msg"
End Function

' ===========================
' LOGGING
' ===========================
Private Sub InitLog()
    Dim f As Integer
    On Error GoTo EH
    EnsureFolderExists GetParentPath(LOG_PATH)
    f = FreeFile
    Open LOG_PATH For Append As #f
    Print #f, ""
    Close #f
    Exit Sub
EH:
    ' swallow
End Sub

Private Sub LogLine(ByVal s As String)
    Dim f As Integer
    On Error Resume Next
    f = FreeFile
    Open LOG_PATH For Append As #f
    Print #f, Format$(Now, "yyyy-mm-dd hh:nn:ss"); " | "; s
    Close #f
End Sub
