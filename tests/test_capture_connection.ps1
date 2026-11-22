$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$source = Join-Path $project 'AstroCam-Capture.ps1'
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($source, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
$function = $ast.Find({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-Connection'
}, $true)
Invoke-Expression $function.Extent.Text
$script:adb = Join-Path $PSScriptRoot 'mock_adb.ps1'

function Assert-ConnectionFailure {
    $script:device = ''
    $failed = $false
    try { Test-Connection } catch { $failed = $true }
    if (-not $failed) { throw 'Expected ambiguous or unauthorized devices to fail.' }
}

$global:MockAdbDevices = @('List of devices attached', "S10E-TEST`tdevice")
$script:device = ''
Test-Connection
if ($script:device -ne 'S10E-TEST') { throw 'Single device was not discovered.' }
$global:MockAdbDevices = @('List of devices attached', "PHONE-A`tdevice", "PHONE-B`tdevice")
Assert-ConnectionFailure
$script:device = 'EXPLICIT-S10E'
Test-Connection
if ($global:MockAdbSerial -ne 'EXPLICIT-S10E') { throw 'Explicit serial was not used.' }
$global:MockAdbDevices = @('List of devices attached', "S10E-TEST`tunauthorized")
Assert-ConnectionFailure
$global:MockAdbDevices = @('List of devices attached')
Assert-ConnectionFailure
Write-Host 'CAPTURE CONNECTION TESTS PASSED'
