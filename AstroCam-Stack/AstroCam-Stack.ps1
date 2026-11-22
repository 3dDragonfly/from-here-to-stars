param([string]$InputFolder = '', [string]$SelectionReport = '', [switch]$Legacy)
$ErrorActionPreference = 'Stop'
try {
    Add-Type -AssemblyName System.Windows.Forms
    $pythonCandidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:USERPROFILE 'AppData\Local\Programs\Python\Python313\python.exe'),
        (Join-Path $env:USERPROFILE 'AppData\Local\Programs\Python\Python312\python.exe')
    )
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notmatch '\\WindowsApps\\') {
        $pythonCandidates += $pythonCommand.Source
    }
    $python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $python) { throw 'Python blev ikke fundet. Installer Python og pakkerne i requirements.txt.' }
    if (-not $InputFolder) {
        $picker = New-Object System.Windows.Forms.FolderBrowserDialog
        $picker.Description = 'Vaelg EN optageserie med DNG-filer (RAW)'
        $picker.ShowNewFolderButton = $false
        $root = Join-Path (Split-Path $PSScriptRoot -Parent) 'AstroCam-billeder'
        if (Test-Path -LiteralPath $root) { $picker.SelectedPath = $root }
        try {
            if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
            $InputFolder = $picker.SelectedPath
        } finally { $picker.Dispose() }
    }
    $stackArguments = @((Join-Path $PSScriptRoot 'astrocam_stack.py'), '--input', $InputFolder)
    if ($SelectionReport) { $stackArguments += @('--selection-report', $SelectionReport) }
    if ($Legacy) { $stackArguments += '--legacy' }
    & $python @stackArguments
    if ($LASTEXITCODE -ne 0) { throw "Stacking fejlede (kode $LASTEXITCODE). Se fejlteksten ovenfor og rapporten i Stack-mappen." }
    Write-Host 'Resultatet ligger i en ny Stack-mappe i optageserien.' -ForegroundColor Green
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
