# Tests for scripts/install.ps1 (the Windows entry).
#
# PLAIN pwsh script (no Pester): Pester is not installed on the dev host, so this
# file dot-sources the four PURE functions from scripts/install.ps1 and asserts
# against fixture strings only. It never calls wsl/docker/podman/python3.
#
# Run:  pwsh -NoProfile -File tests/wsl_gateway.Tests.ps1
# Exit: 0 with "N assertions passed" on success; 1 with "FAIL: <what>" on the
#       first failing assertion.
#
# The dot-source is safe: install.ps1 defines its pure functions first and guards
# the imperative body behind `if ($MyInvocation.InvocationName -ne '.')`, so
# loading the file here (dot-source => InvocationName '.') only defines the
# functions and does NOT run the body (which needs a real Windows + WSL2).

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Locate and load scripts/install.ps1 (relative to THIS file, so it works from
# any working directory).
# ---------------------------------------------------------------------------
$testDir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$installPath  = Join-Path $testDir '../scripts/install.ps1'
if (-not (Test-Path $installPath)) {
  Write-Host "FAIL: scripts/install.ps1 not found (expected at $installPath) -- the functions do not exist yet"
  exit 1
}
$installResolved = (Resolve-Path $installPath).Path
. $installResolved

# The four pure functions must now be available in scope.
foreach ($fn in 'ConvertTo-DistroTable', 'Select-Distro', 'Test-Gateway', 'Resolve-Command') {
  if (-not (Get-Command $fn -ErrorAction SilentlyContinue)) {
    Write-Host "FAIL: function $fn is not defined by scripts/install.ps1"
    exit 1
  }
}

# ---------------------------------------------------------------------------
# Tiny assertion helpers (print and exit 1 on the first failure; count passes).
# ---------------------------------------------------------------------------
$script:AssertCount = 0

function Assert-Equal($Actual, $Expected, $What) {
  $script:AssertCount++
  if ($Actual -ne $Expected) {
    Write-Host "FAIL: $What (expected [$Expected], got [$Actual])"
    exit 1
  }
  Write-Host "  ok: $What"
}

function Assert-True($Condition, $What) {
  $script:AssertCount++
  if (-not $Condition) {
    Write-Host "FAIL: $What"
    exit 1
  }
  Write-Host "  ok: $What"
}

function Assert-Contains($Haystack, $Needle, $What) {
  $script:AssertCount++
  if (-not ([string]$Haystack -like "*$Needle*")) {
    Write-Host "FAIL: $What (expected text to contain [$Needle], was [$Haystack])"
    exit 1
  }
  Write-Host "  ok: $What"
}

# ---------------------------------------------------------------------------
# Fixtures: a realistic `wsl -l -v` block (STATE column first, NAME second; a
# leading '*' marks the default distro; one distro is Not Installed).
# ---------------------------------------------------------------------------
$wslListView = @"
  STATE          NAME
  * Running      Ubuntu
    Stopped      Debian
    Not Installed Alpine
"@

# ---- ConvertTo-DistroTable -------------------------------------------------
Write-Host "ConvertTo-DistroTable"
$table = @(ConvertTo-DistroTable $wslListView)
Assert-Equal $table.Count 3 'parses a 3-distro wsl -l -v block into 3 rows'
Assert-Equal $table[0].Name  'Ubuntu'  'row 0 name is Ubuntu'
Assert-Equal $table[0].State 'Running' 'row 0 state is Running (default, leading * stripped)'
Assert-Equal $table[1].Name  'Debian'  'row 1 name is Debian'
Assert-Equal $table[1].State 'Stopped' 'row 1 state is Stopped'
Assert-Equal $table[2].Name  'Alpine'  'row 2 name is Alpine'
Assert-Equal $table[2].State 'Not Installed' 'row 2 state is Not Installed (two-word state kept)'

$emptyOut = @(ConvertTo-DistroTable '')
Assert-Equal $emptyOut.Count 0 'empty input returns no rows'
$nullOut = @(ConvertTo-DistroTable $null)
Assert-Equal $nullOut.Count 0 'null input returns no rows'

# ---- Select-Distro ---------------------------------------------------------
Write-Host "Select-Distro"
Assert-Equal (Select-Distro $table 'WindowsServer') 'WindowsServer' 'explicit distro wins even when not in the table'
Assert-Equal (Select-Distro $table 'Debian')        'Debian'        'explicit distro returns itself'
Assert-Equal (Select-Distro $table '')              'Ubuntu'        'no explicit -> first Running/Stopped in wsl -l -v order'
# A table whose first row is Not Installed must be skipped.
$mixed = @(ConvertTo-DistroTable "  STATE          NAME`n    Not Installed Alpine`n    Running      Fedora")
Assert-Equal (Select-Distro $mixed '') 'Fedora' 'skips Not Installed rows and picks the first Running/Stopped'
$onlyNotInstalled = @(ConvertTo-DistroTable "  STATE          NAME`n    Not Installed Alpine")
Assert-Equal (Select-Distro $onlyNotInstalled '') '' 'only Not Installed distros -> empty string'

# ---- Test-Gateway ----------------------------------------------------------
Write-Host "Test-Gateway"
$g1 = Test-Gateway $false '' $true $true $true
Assert-Equal      $g1.Action 'abort'        'no WSL2 -> abort'
Assert-Contains   $g1.Message 'WSL2'        'no-WSL2 message names WSL2'
Assert-Contains   $g1.Fix 'wsl --install'   'no-WSL2 fix is wsl --install (admin + reboot)'

$g2 = Test-Gateway $true '' $true $true $true
Assert-Equal $g2.Action 'abort' 'WSL2 present but no usable distro -> abort'

$g3 = Test-Gateway $true 'Ubuntu' $false $false $true
Assert-Equal      $g3.Action 'abort'            'no docker AND no podman -> abort'
Assert-Contains   $g3.Message 'Docker Desktop'  'no-runtime message names Docker Desktop (off)'
Assert-Contains   $g3.Message 'WSL'             'no-runtime message names the WSL integration'
Assert-Contains   $g3.Message 'podman'          'no-runtime message names podman (not installed)'

$g4 = Test-Gateway $true 'Ubuntu' $true $false $false
Assert-Equal $g4.Action 'install-python' 'runtime ok but python3 missing -> install-python'

$g5 = Test-Gateway $true 'Ubuntu' $true $false $true
Assert-Equal $g5.Action 'delegate' 'all checks pass (docker ok) -> delegate'

$g6 = Test-Gateway $true 'Ubuntu' $false $true $true
Assert-Equal $g6.Action 'delegate' 'podman present (docker absent) is still a runtime -> delegate'

# ---- Resolve-Command -------------------------------------------------------
Write-Host "Resolve-Command"
foreach ($c in 'status', 'up', 'down', 'remove') {
  $r = @(Resolve-Command $c)
  Assert-Equal ($r -join ' ') "qctx stack $c" "Resolve-Command '$c' -> qctx stack $c"
}
Assert-True ($null -eq (Resolve-Command 'install')) "Resolve-Command 'install' -> null"
Assert-True ($null -eq (Resolve-Command 'bogus'))   "Resolve-Command 'bogus' -> null"
Assert-True ($null -eq (Resolve-Command ''))        "Resolve-Command '' -> null"

# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "$($script:AssertCount) assertions passed"
exit 0
