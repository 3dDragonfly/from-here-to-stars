param([string]$Device = '')

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$script:adb = Join-Path $PSScriptRoot 'adb.exe'
$script:device = $Device
$script:cameraPath = '/sdcard/DCIM/Camera'
$script:shot = 0
$script:total = 0
$script:before = @{}
$script:destination = $null

$form = New-Object System.Windows.Forms.Form
$form.Text = 'From Here to Stars - S10e Capture'
$form.Size = New-Object System.Drawing.Size(460, 365)
$form.StartPosition = 'CenterScreen'

$label = New-Object System.Windows.Forms.Label
$label.Text = 'Samsung Galaxy S10e - AstroCam'
$label.SetBounds(20, 20, 410, 30)
$form.Controls.Add($label)

function Add-Number($title, $y, $value, $max) {
    $l = New-Object System.Windows.Forms.Label
    $l.Text = $title
    $l.SetBounds(20, $y, 150, 25)
    $form.Controls.Add($l)
    $n = New-Object System.Windows.Forms.NumericUpDown
    $n.SetBounds(180, $y, 100, 25)
    $n.Minimum = 1
    $n.Maximum = $max
    $n.Value = $value
    $form.Controls.Add($n)
    return $n
}

$count = Add-Number 'Antal billeder' 65 5 1000
$exposure = Add-Number 'Lukketid (sek.)' 100 10 300
$pause = Add-Number 'Pause (sek.)' 135 3 60

$connect = New-Object System.Windows.Forms.Button
$connect.Text = 'Test forbindelse'
$connect.SetBounds(20, 185, 120, 35)
$form.Controls.Add($connect)
$start = New-Object System.Windows.Forms.Button
$start.Text = 'START'
$start.SetBounds(150, 185, 100, 35)
$form.Controls.Add($start)
$stop = New-Object System.Windows.Forms.Button
$stop.Text = 'STOP'
$stop.SetBounds(260, 185, 100, 35)
$stop.Enabled = $false
$form.Controls.Add($stop)
$status = New-Object System.Windows.Forms.Label
$status.Text = 'Klar'
$status.SetBounds(20, 245, 415, 65)
$form.Controls.Add($status)
$timer = New-Object System.Windows.Forms.Timer

function Test-Connection {
    if (-not (Test-Path -LiteralPath $script:adb)) { throw 'adb.exe mangler i programmets mappe' }
    if (-not $script:device) {
        $listing = @(& $script:adb devices 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "ADB device discovery failed: $listing" }
        $serials = @($listing | ForEach-Object {
            if ("$_" -match '^([^\s]+)\s+device$') { $Matches[1] }
        })
        if ($serials.Count -ne 1) {
            throw 'Connect and authorize one S10e, or start with -Device <ADB serial or IP:port>.'
        }
        $script:device = $serials[0]
    }
    $result = & $script:adb -s $script:device get-state 2>&1
    if ($LASTEXITCODE -ne 0 -or "$result".Trim() -ne 'device') {
        throw 'Telefonen er ikke forbundet'
    }
}

function Test-ProMode {
    $dump = & $script:adb -s $script:device shell uiautomator dump /sdcard/window.xml 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Kunne ikke hente kameradata: $dump" }
    $rawXml = (& $script:adb -s $script:device shell cat /sdcard/window.xml 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw 'Kunne ikke læse kameradata' }
    try { [xml]$doc = $rawXml } catch { throw 'Ugyldige XML-data fra kameraet' }
    $mode = $doc.SelectSingleNode("//*[@resource-id='com.sec.android.app.camera:id/shooting_mode_list']")
    if ($null -eq $mode) { throw 'Kameratilstand kunne ikke aflæses. Åbn kameraet på telefonen.' }
    $description = $mode.GetAttribute('content-desc')
    if ($description -ne 'Pro, Mode') { throw "PRO er ikke aktiv. Aktuel tilstand: $description" }
}

function Get-CameraFiles {
    $entries = & $script:adb -s $script:device shell ls -1 $script:cameraPath 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Kameramappen kunne ikke læses: $entries" }
    return @($entries | ForEach-Object { "$($_)".Trim().TrimEnd("`r") } |
        Where-Object { $_ -match '\.(jpe?g|dng|heic)$' })
}

function Save-NewPhotos {
    $files = @(Get-CameraFiles | Where-Object { -not $script:before.ContainsKey($_) })
    $saved = 0
    foreach ($name in $files) {
        $remote = "$script:cameraPath/$name"
        $output = & $script:adb -s $script:device pull $remote $script:destination 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Kunne ikke hente $name`: $output" }
        $saved++
    }
    return $saved
}

function Stop-Series($message) {
    $timer.Stop()
    $start.Enabled = $true
    $stop.Enabled = $false
    $status.Text = $message
}

function Take-Shot {
    $script:shot++
    $status.Text = "Billede $script:shot af $script:total - eksponerer"
    $form.Refresh()
    $result = & $script:adb -s $script:device shell input keyevent 27 2>&1
    if ($LASTEXITCODE -ne 0) {
        Stop-Series "ADB-fejl: $result"
        return
    }
    $timer.Start()
}

$connect.Add_Click({
    try {
        Test-Connection
        Test-ProMode
        $status.Text = 'Forbundet; PRO er aktiv'
    } catch { $status.Text = $_.Exception.Message }
})

$start.Add_Click({
    try {
        Test-Connection
        Test-ProMode
        $script:before = @{}
        foreach ($name in @(Get-CameraFiles)) { $script:before[$name] = $true }
        $parent = Join-Path $PSScriptRoot 'AstroCam-billeder'
        $script:destination = Join-Path $parent (Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')
        [void](New-Item -ItemType Directory -Path $script:destination -Force)
        $script:shot = 0
        $script:total = [int]$count.Value
        $seconds = [int]$exposure.Value + [int]$pause.Value
        $timer.Interval = $seconds * 1000
        $start.Enabled = $false
        $stop.Enabled = $true
        Take-Shot
    } catch { $status.Text = $_.Exception.Message }
})

$timer.Add_Tick({
    $timer.Stop()
    if ($script:shot -lt $script:total) {
        Take-Shot
    } else {
        try {
            $status.Text = 'Henter billeder fra telefonen...'
            $form.Refresh()
            $saved = Save-NewPhotos
            Stop-Series "Færdig: $saved nye filer hentet til $script:destination"
        } catch { Stop-Series "Billeder taget, men overførsel fejlede: $($_.Exception.Message)" }
    }
})

$stop.Add_Click({ Stop-Series 'Stoppet af bruger; billeder er stadig på telefonen' })
$form.Add_FormClosing({ $timer.Stop() })
[void]$form.ShowDialog()
