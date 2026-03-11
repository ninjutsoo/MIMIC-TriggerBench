<#
.SYNOPSIS
  Download MIMIC-IV 3.1 from PhysioNet with multiple files in parallel.
  Works on Windows PowerShell 5.1.

.DESCRIPTION
  Runs 3 wget processes at a time in the foreground so progress streams to the terminal.
  Resume (-c) is used so you can stop and rerun.
  Set credentials before running:
    $env:PHYSIONET_USER = "ninjutsoo"
    $env:PHYSIONET_PASS = "your_password"
#>

param(
    [int] $ParallelJobs = 3,
    [string] $BaseUrl = "https://physionet.org/files/mimiciv/3.1",
    [string] $Sha256SumsPath = "physionet.org\files\mimiciv\3.1\SHA256SUMS.txt",
    [string] $WgetPath = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\JernejSimoncic.Wget_Microsoft.Winget.Source_8wekyb3d8bbwe\wget.exe"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path -Parent
Set-Location $root

if (-not $env:PHYSIONET_USER -or -not $env:PHYSIONET_PASS) {
    Write-Warning "Set PHYSIONET_USER and PHYSIONET_PASS env vars first."
    exit 1
}

if (-not (Test-Path $WgetPath)) {
    Write-Warning "wget not found at: $WgetPath"
    exit 1
}

$lines = Get-Content $Sha256SumsPath -ErrorAction SilentlyContinue
if (-not $lines) {
    Write-Warning "SHA256SUMS not found at $Sha256SumsPath."
    exit 1
}

$relPaths = $lines | ForEach-Object {
    $part = ($_ -split " ", 2)[1]
    if ($part -and $part -match "/") { $part.Trim() }
}

$urls = $relPaths | ForEach-Object { "$BaseUrl/$_" }
Write-Host "Total files: $($urls.Count). Running $ParallelJobs in parallel."

$wgetExe = $WgetPath
$user = $env:PHYSIONET_USER
$pass = $env:PHYSIONET_PASS

$rootAbs = (Get-Location).Path
$queue = [System.Collections.Queue]::new($urls)
$active = [System.Collections.ArrayList]::new()

while ($queue.Count -gt 0 -or $active.Count -gt 0) {
    while ($active.Count -lt $ParallelJobs -and $queue.Count -gt 0) {
        $url = $queue.Dequeue()
        $args = @('-c','-N','-x','-P',$rootAbs,'--tries=0','--waitretry=5','--read-timeout=30','--timeout=30',
                  '--user',$user,'--password',$pass,$url)
        $proc = Start-Process -FilePath $wgetExe -ArgumentList $args -WorkingDirectory $rootAbs `
            -NoNewWindow -PassThru
        [void]$active.Add([PSCustomObject]@{ Process = $proc; Url = $url })
    }
    if ($active.Count -eq 0) { break }
    do {
        Start-Sleep -Milliseconds 500
        $idx = -1
        for ($i = 0; $i -lt $active.Count; $i++) {
            if ($active[$i].Process.HasExited) { $idx = $i; break }
        }
    } while ($idx -lt 0)
    $active.RemoveAt($idx)
}

Write-Host "Done."
