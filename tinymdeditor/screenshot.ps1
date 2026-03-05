param([string]$OutFile, [string]$ExePath, [string]$ArgStr)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WinAPI {
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, uint nFlags);
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

if ($ArgStr -and $ArgStr.Length -gt 0) {
    $proc = Start-Process -FilePath $ExePath -ArgumentList $ArgStr -PassThru
} else {
    $proc = Start-Process -FilePath $ExePath -PassThru
}
Start-Sleep -Milliseconds 4000
$proc.Refresh()

if ($proc.MainWindowHandle -eq [IntPtr]::Zero) {
    Write-Host "ERROR: No main window handle"
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

$rect = New-Object WinAPI+RECT
[WinAPI]::GetWindowRect($proc.MainWindowHandle, [ref]$rect) | Out-Null

$w = $rect.Right - $rect.Left
$h = $rect.Bottom - $rect.Top
Write-Host "Window rect: $w x $h at ($($rect.Left), $($rect.Top))"

$bmp = New-Object System.Drawing.Bitmap($w, $h)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $g.GetHdc()
# PW_RENDERFULLCONTENT = 2 captures even if occluded
[WinAPI]::PrintWindow($proc.MainWindowHandle, $hdc, 2) | Out-Null
$g.ReleaseHdc($hdc)
$g.Dispose()
$bmp.Save($OutFile, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()

Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Write-Host "Saved: $OutFile"
