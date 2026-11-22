# Fixture only: no real phone connection or camera commands.
$global:LASTEXITCODE = 0
if ($args.Count -eq 1 -and $args[0] -eq 'devices') {
    $global:MockAdbDevices
} elseif ($args.Count -eq 3 -and $args[0] -eq '-s' -and $args[2] -eq 'get-state') {
    $global:MockAdbSerial = $args[1]
    'device'
} else {
    $global:LASTEXITCODE = 1
    'Unexpected mock ADB invocation'
}
