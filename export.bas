' ===== ArchiveBackup.bas =====
' Outlook VBA: Export Online Archive to .MSG with checkpoint/restart & logging
' Fixes:
'  - All Dim statements moved before executable code (VBA requirement)
'  - Safe filename/path builder to prevent STG_E_INVALIDNAME (-2147286788)

Option Explicit

' ---------------------------
' CONFIG — EDIT THESE
' ---------------------------
Private Const ROOT_EXPORT As String = "D:\MailBackup"                ' Root backup folder
Private Const LOG_PATH As String = "D:\MailBackup\backup.log"        ' Log file
Private Const CHECKPOINT_CSV As String = "D:\MailBackup\checkpoint.csv" ' Checkpoint file

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

' ---------------------------
' ENTRY POINT
' ---------------------------
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

' ---------------------------
' STORE / FOLDER WALK
' ---------------------------
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

' ---------------------------
' EXPORT ITEMS
' ---------------------------
Private Sub ExportFolderItems(ByVal fld As Outlook.Folder, ByVal targetPath As String, ByVal storeId As String)
    Dim itms As Outlook.Items
    Dim i As Long
    Dim it As Object

    On Error GoTo EH

    Set itms = fld.Items
    On Error Resume Next
    itms.Sort "[ReceivedTime]", False
    itms.IncludeRecurrences = True
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

' ---------------------------
' CHECKPOINT CSV
' ---------------------------
' CSV columns: StoreID,FolderID,EntryID,LastModUTC,FilePath
Private Function HasItemChangedSinceCheckpoint(ByVal storeId As String, ByVal folderId As String, _
                                               ByVal entryId As String, ByVal lastMod As Date, _
                                               ByVal filePath As String) As Boolean
    Dim f As Integer
    Dim line As String
    Dim found As Boolean
    Dim lastModStr As String
    Dim parts() As String
    Dim cpLast As String

    If Not FileExists(CHECKPOINT_CSV) Then
        HasItemChangedSinceCheckpoint = True
        Exit Function
    End If

    f = FreeFile
    On Error GoTo EH
    Open CHECKPOINT_CSV For Input As #f

    lastModStr = Format$(lastMod, "yyyy-mm-dd hh:nn:ss")
    found = False

    Do While Not EOF(f)
