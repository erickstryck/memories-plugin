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

# ---- Imperative body (host-faked) -----------------------------------------
# The pure-function assertions above CANNOT see the body's control flow, and two
# real regressions lived there: a `$null = wsl ...` that swallowed the whole
# journey's stdout (the progress bar and the final summary with the URLs), and
# the install-python branch ending without delegating. Neither is visible to
# Test-Gateway, so this section drives the REAL script end-to-end with shimmed
# `where.exe` / `wsl` on PATH. It self-skips when bash is unavailable (the fakes
# are bash scripts), so a pwsh-only host is not broken by it.
$bashForBody = Get-Command bash -ErrorAction SilentlyContinue
$pwshBin     = Join-Path $PSHOME 'pwsh'
if (-not $bashForBody -or -not (Test-Path $pwshBin)) {
  Write-Host "(body test skipped: needs bash + a nested pwsh to shim where.exe/wsl)"
} else {
  $fakeDir = Join-Path ([System.IO.Path]::GetTempPath()) ("qctx-body-" + [guid]::NewGuid().ToString("N"))
  New-Item -ItemType Directory $fakeDir | Out-Null
  $env:QCTX_FAKE_DIR = $fakeDir
  try {
    # The fakes read their exit codes from sidecar .rc files, so one fake set
    # serves every scenario. Written by bash (LF, working shebang).
    $bashBlock = @'
D="$QCTX_FAKE_DIR"
cat > "$D/where.exe" <<'EOS'
#!/usr/bin/env bash
exit 0
EOS
cat > "$D/wsl" <<'EOS'
#!/usr/bin/env bash
D="$(cd "$(dirname "$0")" && pwd)"
if [ "$1" = "-l" ]; then printf '  STATE          NAME\n  * Running      Ubuntu\n'; exit 0; fi
shift; shift
[ "$1" = "--" ] && shift
case "$*" in
  *command\ -v\ python3*) exit $(cat "$D/python3.rc") ;;
  *command\ -v\ docker*)  exit 0 ;;
  *apt-get*)              exit $(cat "$D/apt.rc") ;;
  *"-lc qctx stack status"*) echo "BODY_IN_DISTRO_STATUS"; exit $(cat "$D/cmd.rc") ;;
  *install.sh*)           echo "BODY_WIZARD_STREAMED"; exit 0 ;;
  *)                      exit 0 ;;
esac
EOS
chmod +x "$D/where.exe" "$D/wsl"
echo 0 > "$D/python3.rc"; echo 0 > "$D/apt.rc"; echo 0 > "$D/cmd.rc"
'@
    & $bashForBody.Source -c $bashBlock
    if ($LASTEXITCODE -ne 0) { throw "the fake wsl/where harness failed to set up (bash)" }

    $sep  = if ($IsLinux -or $IsMacOS) { ":" } else { ";" }
    $savedPath = $env:PATH
    $env:PATH  = $fakeDir + $sep + $savedPath

    # S2: a distro without python3, the user consents -> the body installs it
    # and CONTINUES into the wizard (the fall-through the switch used to drop),
    # and the wizard output streams to the console (is not captured).
    Set-Content (Join-Path $fakeDir 'python3.rc') -Value "1"
    $out  = "y`n" | & $pwshBin -NoProfile -File $installResolved --stack auto 2>&1
    $code = $LASTEXITCODE
    Assert-True     ($code -eq 0)         "body: the install-python path exits 0 (consented)"
    Assert-Contains $out "starting the wizard"  "body: after installing python3 it starts the wizard"
    Assert-Contains $out "BODY_WIZARD_STREAMED" "body: the wizard output streams (not captured)"

    # S3: -Command status with an in-distro exit 3 -> the script exits 3 and
    # streams the in-distro output (spec rule 5, no stdout capture).
    Set-Content (Join-Path $fakeDir 'python3.rc') -Value "0"
    Set-Content (Join-Path $fakeDir 'cmd.rc')     -Value "3"
    $out2  = & $pwshBin -NoProfile -File $installResolved -Command status 2>&1
    $code2 = $LASTEXITCODE
    Assert-True     ($code2 -eq 3)        "body: -Command propagates the in-distro exit code"
    Assert-Contains $out2 "BODY_IN_DISTRO_STATUS" "body: -Command output streams (not captured)"
  } finally {
    $env:PATH = $savedPath
    Remove-Item Env:QCTX_FAKE_DIR -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $fakeDir -ErrorAction SilentlyContinue
  }
}

# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "$($script:AssertCount) assertions passed"
exit 0
