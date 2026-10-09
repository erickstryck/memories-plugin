# install.ps1 - the Windows entry for the qctx stack wizard.
#
# It verifies and delegates; it decides nothing. install.sh stays the only
# decision-maker (spec rule 1): past the `wsl -d ... -- bash`, the journey is
# identical on every platform.
#
#   .\install.ps1                    # auto-pick a distro, run the wizard
#   .\install.ps1 --stack auto       # pass any wizard arg straight to install.sh
#   .\install.ps1 -Distro Ubuntu     # pin a specific WSL distro
#   .\install.ps1 -Command status    # run `qctx stack status` inside the distro
#                                    # (decision 24) - does NOT re-run the wizard
#
# The script's exit code is the exit code of the process run inside the distro
# (spec rule 5): a wizard abort is not turned into a generic PowerShell error.

param(
  # A specific WSL distro to target. When empty, the first Running/Stopped
  # distro (in `wsl -l -v` order, never "Not Installed") is chosen.
  [Parameter()] [string]$Distro = '',

  # A qctx stack sub-command (status|up|down|remove) to run inside the distro
  # instead of the wizard. Any other value is rejected (Resolve-Command).
  [Parameter()] [string]$Command = '',

  # Every remaining argument (e.g. `--stack auto`) is forwarded verbatim to
  # install.sh.
  [Parameter(ValueFromRemainingArguments = $true)] [string[]]$WizardArgs = @()
)

# ===========================================================================
# PURE FUNCTIONS (no side effects; no wsl/docker/podman/python3 calls inside).
# Defined first so a test can dot-source this file and exercise them in
# isolation, without running the imperative body below.
# ===========================================================================

function ConvertTo-DistroTable {
  <#
  .SYNOPSIS
    Parse the output of `wsl -l -v` into rows of (Name, State).
  .DESCRIPTION
    A realistic `wsl -l -v` block has a header line (STATE first, NAME second)
    then one line per distro:
        STATE          NAME
        * Running      Ubuntu
          Stopped      Debian
          Not Installed Alpine
    The leading '*' (plus its whitespace) marks the default distro and is
    stripped. The state is one of "Running", "Stopped", or "Not Installed"
    (two words); the remainder of the line is the name.
  .PARAMETER wslListView
    The full multi-line text of `wsl -l -v` (may be empty or $null).
  .OUTPUTS
    [pscustomobject[]] - one row per distro with .Name and .State. Empty for
    empty/absent input.
  #>
  param([string]$wslListView)
  $rows = @()
  if ([string]::IsNullOrWhiteSpace($wslListView)) { return [pscustomobject[]]$rows }

  foreach ($line in ($wslListView -split "`r?`n")) {
    # Drop the leading whitespace that indents every row (and the header).
    $body = $line.TrimStart(' ', "`t")
    if ($body.Length -eq 0) { continue }
    # Skip the header line (STATE ... NAME) and any empty remainder.
    if ($body -ceq 'STATE' -or $body -clike 'STATE*') { continue }
    # Strip the default-distro marker '*' (immediately after the indent).
    if ($body.StartsWith('*')) { $body = $body.Substring(1).TrimStart(' ', "`t") }
    if ($body.Length -eq 0) { continue }

    # Split into tokens: the state is 1 or 2 words, the name is the rest.
    $tokens = @($body -split '\s+' | Where-Object { $_ -ne '' })
    if ($tokens.Count -lt 2) { continue }
    if ($tokens[0] -eq 'Not' -and $tokens[1] -eq 'Installed') {
      $state = 'Not Installed'
      $name  = ($tokens | Select-Object -Skip 2) -join ' '
    } else {
      $state = $tokens[0]
      $name  = ($tokens | Select-Object -Skip 1) -join ' '
    }
    if ($name.Length -eq 0) { continue }
    $rows += [pscustomobject]@{ Name = $name; State = $state }
  }
  return [pscustomobject[]]$rows
}

function Select-Distro {
  <#
  .SYNOPSIS
    Choose the distro to target from a parsed `wsl -l -v` table.
  .DESCRIPTION
    If $explicit is non-empty it wins (returned as-is, even when it is not in
    the table). Otherwise the first distro whose State is "Running" or
    "Stopped", in `wsl -l -v` order, is returned. A distro whose State is
    "Not Installed" is never chosen. If none qualifies, returns ''.
  .PARAMETER table
    Output of ConvertTo-DistroTable.
  .PARAMETER explicit
    An explicit distro name (e.g. from -Distro); '' to auto-pick.
  #>
  param($table, [string]$explicit)
  if ($null -ne $explicit -and $explicit -ne '') { return $explicit }
  foreach ($row in $table) {
    if ($row.State -eq 'Running' -or $row.State -eq 'Stopped') { return $row.Name }
  }
  return ''
}

function Test-Gateway {
  <#
  .SYNOPSIS
    Decide, from verified facts, what the Windows entry must do next.
  .DESCRIPTION
    Returns a [pscustomobject] with .Action ('abort' | 'install-python' |
    'delegate') plus a human .Message and a .Fix. Decisions, in priority
    order:
      1. WSL2 missing           -> abort (wsl --install; admin + reboot).
      2. No usable distro       -> abort.
      3. No docker AND no podman-> abort (names the three likely causes).
      4. python3 missing        -> install-python (apt, with consent).
      5. all good               -> delegate.
    A present docker OR podman satisfies the runtime check (step 3).
  #>
  param(
    [bool]$wslPresent,
    [string]$distro,
    [bool]$dockerOk,
    [bool]$podmanOk,
    [bool]$python3Ok
  )

  if (-not $wslPresent) {
    return [pscustomobject]@{
      Action  = 'abort'
      Message = "WSL2 is missing: no WSL on this host, so there is no Linux distro to install the stack in."
      Fix     = "wsl --install   (run from an Administrator PowerShell; it needs a reboot to finish)"
    }
  }

  if ([string]::IsNullOrWhiteSpace($distro)) {
    return [pscustomobject]@{
      Action  = 'abort'
      Message = "WSL2 is present but no usable distro was found (none Running or Stopped; only 'Not Installed' entries are skipped)."
      Fix     = "Install or start a distro first, e.g. `wsl --install -d Ubuntu`, then re-run."
    }
  }

  if (-not $dockerOk -and -not $podmanOk) {
    return [pscustomobject]@{
      Action  = 'abort'
      Message = "No container runtime inside the '$distro' distro: neither `docker` nor `podman` answered. Likely causes: (1) Docker Desktop is off, (2) the WSL integration for '$distro' is unchecked in Docker Desktop > Settings > Resources, or (3) podman is not installed."
      Fix     = "Start Docker Desktop and enable WSL integration for '$distro' (or install podman), then re-run."
    }
  }

  if (-not $python3Ok) {
    return [pscustomobject]@{
      Action  = 'install-python'
      Message = "python3 is missing inside the '$distro' distro (the wizard needs it)."
      Fix     = "apt install python3"
    }
  }

  return [pscustomobject]@{
    Action  = 'delegate'
    Message = "All checks pass inside '$distro': a container runtime answers and python3 is present."
    Fix     = ''
  }
}

function Resolve-Command {
  <#
  .SYNOPSIS
    Map a stack sub-command to the `qctx stack <cmd>` argv, or reject it.
  .DESCRIPTION
    'status'/'up'/'down'/'remove' -> @('qctx','stack',<cmd>) (decision 24: run
    the launcher inside the distro). Any other input (including 'install',
    which belongs to install.sh, not the entry) -> $null.
  #>
  param([string]$command)
  switch ($command) {
    'status' { return @('qctx', 'stack', 'status') }
    'up'     { return @('qctx', 'stack', 'up') }
    'down'   { return @('qctx', 'stack', 'down') }
    'remove' { return @('qctx', 'stack', 'remove') }
    default  { return $null }
  }
}

# ===========================================================================
# IMPERATIVE BODY.
#
# Guard: when this file is DOT-SOURCED (tests), $MyInvocation.InvocationName
# is '.', so the body is skipped and only the pure functions above are defined.
# When run as the entry point (`pwsh install.ps1` / `./install.ps1` / `bash -c`
# ...), InvocationName is the script path, so the body runs. This is what lets
# tests/wsl_gateway.Tests.ps1 load the functions without touching wsl/docker.
#
# The body is written as top-level code under the dot-source guard, so the
# entry script reads as a straight-line flow. Each terminal branch ends the
# script with the exit code it names - the in-distro $LASTEXITCODE for
# delegation, 1 for an abort - per spec rule 5; a native call's exit code does
# not by itself become the script's (measured: a script ending right after a
# failed native call still exits 0).
# ===========================================================================
if ($MyInvocation.InvocationName -ne '.') {

  # Resolve the clone root from this file's location (scripts/..), so install.sh
  # is always the one from the same copy the user cloned.
  $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
  $root      = Split-Path -Parent $scriptDir
  $installSh = Join-Path $scriptDir 'install.sh'

  # --- Verification 1: is WSL2 even present? (where.exe wsl) ---------------
  # Capture the exit code right after the native call so nothing clobbers it.
  $null = where.exe wsl 2>&1
  $wslPresent = ($LASTEXITCODE -eq 0)

  $distroName   = ''
  $dockerOk     = $false
  $podmanOk     = $false
  $python3Ok    = $false

  if ($wslPresent) {
    # --- Verification 2: which distro, and what exists inside it? ----------
    $wslList = (wsl -l -v 2>&1) | Out-String
    $table   = @(ConvertTo-DistroTable $wslList)
    $distroName = Select-Distro $table $Distro

    if ($distroName -ne '') {
      # Probe each binary inside the distro; a silent/present-but-unreachable
      # binary counts as absent. Success = both the binary exists AND info works.
      $null = wsl -d $distroName -- sh -lc 'command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1' 2>&1
      $dockerOk  = ($LASTEXITCODE -eq 0)
      $null = wsl -d $distroName -- sh -lc 'command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1' 2>&1
      $podmanOk  = ($LASTEXITCODE -eq 0)
      $null = wsl -d $distroName -- sh -lc 'command -v python3 >/dev/null 2>&1' 2>&1
      $python3Ok = ($LASTEXITCODE -eq 0)
    }
  }

  $gateway = Test-Gateway -wslPresent $wslPresent -distro $distroName `
                          -dockerOk $dockerOk -podmanOk $podmanOk -python3Ok $python3Ok

  switch ($gateway.Action) {

    'abort' {
      # The entry verifies and reports; it never installs the heavy dependency.
      Write-Host "install.ps1: $($gateway.Message)"
      Write-Host "  fix: $($gateway.Fix)"
      Write-Host "  (This script only verifies and delegates - it will NOT install that for you.)"
      exit 1
    }

    'install-python' {
      # python3 is the one dependency the entry MAY install - but only with the
      # user's consent (decision 19 / spec rule 3).
      $answer = Read-Host "python3 is missing in '$distroName'. Install it now? [y/N]"
      if ($answer -ne 'y' -and $answer -ne 'Y' -and $answer -ne 'yes') {
        Write-Host "  install.ps1: declined - installing python3 in '$distroName' is required before the wizard can run."
        exit 1
      }
      $null = wsl -d $distroName -- sh -lc 'apt-get update && apt-get install -y python3'
      if ($LASTEXITCODE -ne 0) {
        Write-Host "  install.ps1: `apt install python3` in '$distroName' failed (exit $LASTEXITCODE)."
        exit $LASTEXITCODE
      }
      Write-Host "  install.ps1: python3 installed in '$distroName'."
    }

    'delegate' {
      $stackCmd = Resolve-Command $Command
      if ($stackCmd -ne $null) {
        # -Command: run `qctx stack <cmd>` inside the distro via the launcher
        # (decision 24). The wizard is NOT re-triggered.
        $null = wsl -d $distroName -- bash -lc ('qctx stack ' + $stackCmd[2])
        exit $LASTEXITCODE
      }
      if (-not (Test-Path $installSh)) {
        Write-Host "install.ps1: install.sh not found at $installSh - is the repo fully cloned?"
        exit 1
      }
      # No -Command: hand everything to install.sh, the only decision-maker.
      $null = wsl -d $distroName -- bash $installSh @WizardArgs
      exit $LASTEXITCODE
    }

  }

}
