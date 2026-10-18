[CmdletBinding()]
param([switch]$Neural, [switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
}
$Python = '.venv\Scripts\python.exe'
& $Python -c "import sys; assert sys.version_info[:2] == (3,12), 'Use the tested Python 3.12 environment'"
if ($LASTEXITCODE -ne 0) { throw 'The virtual environment must use Python 3.12.' }
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if ($Neural) {
    & "$PSScriptRoot\install-neural.ps1" -Python $Python -CpuOnly:$CpuOnly
}
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Software tests failed.' }
& $Python app.py --diagnostics
if ($LASTEXITCODE -ne 0) { throw 'Startup diagnostics failed.' }
Write-Host 'Software setup verified. Run .venv\Scripts\python.exe app.py to open NeuroIris.'
