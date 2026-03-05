# Startup benchmark — measures time until MainWindowHandle appears and is visible.

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class BenchAPI {
    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

$exes = @(
    @{ Name = "WebView2"; Path = "webview\tinymd-webview.exe" },
    @{ Name = "GDI";      Path = "gdi\tinymd-gdi.exe" },
    @{ Name = "Direct2D"; Path = "direct2d\tinymd-d2d.exe" }
)

$testFile = "test.md"
$results = @()

function MeasureStartup($exePath, $argStr) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    if ($argStr -and $argStr.Length -gt 0) {
        $proc = Start-Process -FilePath $exePath -ArgumentList $argStr -PassThru
    } else {
        $proc = Start-Process -FilePath $exePath -PassThru
    }

    # Poll until main window is visible with non-zero size
    for ($tick = 0; $tick -lt 3000; $tick++) {
        Start-Sleep -Milliseconds 5
        $proc.Refresh()
        if ($proc.MainWindowHandle -ne [IntPtr]::Zero) {
            if ([BenchAPI]::IsWindowVisible($proc.MainWindowHandle)) {
                $rect = New-Object BenchAPI+RECT
                [BenchAPI]::GetWindowRect($proc.MainWindowHandle, [ref]$rect) | Out-Null
                $w = $rect.Right - $rect.Left
                if ($w -gt 100) { break }
            }
        }
    }
    $sw.Stop()
    $ms = $sw.ElapsedMilliseconds
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    return $ms
}

foreach ($exe in $exes) {
    $coldTimes = @()
    $warmTimes = @()

    Write-Host "Testing $($exe.Name)..."

    # Cold runs (5x)
    for ($i = 0; $i -lt 5; $i++) {
        $ms = MeasureStartup $exe.Path $testFile
        $coldTimes += $ms
        Start-Sleep -Milliseconds 800
    }

    # Warm runs (10x)
    for ($i = 0; $i -lt 10; $i++) {
        $ms = MeasureStartup $exe.Path $testFile
        $warmTimes += $ms
        Start-Sleep -Milliseconds 200
    }

    $coldAvg = [math]::Round(($coldTimes | Measure-Object -Average).Average)
    $coldMin = ($coldTimes | Measure-Object -Minimum).Minimum
    $coldMax = ($coldTimes | Measure-Object -Maximum).Maximum
    $warmSorted = $warmTimes | Sort-Object
    $warmAvg = [math]::Round(($warmTimes | Measure-Object -Average).Average)
    $warmMed = $warmSorted[4]
    $warmMin = ($warmTimes | Measure-Object -Minimum).Minimum
    $warmMax = ($warmTimes | Measure-Object -Maximum).Maximum

    $results += [PSCustomObject]@{
        Name       = $exe.Name
        ColdAvg    = "$coldAvg ms"
        ColdRange  = "$coldMin - $coldMax ms"
        ColdRuns   = ($coldTimes -join ", ") + " ms"
        WarmAvg    = "$warmAvg ms"
        WarmMedian = "$warmMed ms"
        WarmRange  = "$warmMin - $warmMax ms"
        WarmRuns   = ($warmTimes -join ", ") + " ms"
    }

    Write-Host "  Done: cold=$coldAvg ms, warm=$warmAvg ms"
}

Write-Host ""
Write-Host "============================================="
Write-Host "  STARTUP BENCHMARK (window visible + sized)"
Write-Host "  5 cold + 10 warm runs per prototype"
Write-Host "============================================="
Write-Host ""
$results | Format-Table Name, ColdAvg, ColdRange, WarmAvg, WarmMedian, WarmRange -AutoSize
Write-Host ""
Write-Host "--- Raw timings ---"
foreach ($r in $results) {
    Write-Host "$($r.Name):"
    Write-Host "  Cold: $($r.ColdRuns)"
    Write-Host "  Warm: $($r.WarmRuns)"
}
