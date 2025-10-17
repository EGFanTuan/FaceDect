param(
    [switch]$CreateVenv = $true,
    [string]$PythonPath = "python",
    [string]$TorchVersion = "2.5.1",
    [string]$TorchVisionVersion = "0.20.1",
    [string]$TorchAudioVersion = "2.5.1",
    [string]$RequirementsPath = "./requirements.txt"
)

$ErrorActionPreference = "Stop"

function Write-Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[error] $msg" -ForegroundColor Red }
function Exec($cmd, [switch]$Quiet) {
    if (-not $Quiet) { Write-Host "> $cmd" -ForegroundColor DarkGray }
    iex $cmd
}

Write-Section "Checking Python"
try {
    $pyVer = & $PythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Write-Host "Python $pyVer detected"
}
catch {
    Write-Err "Python not found in PATH. Install Python 3.9–3.12 and retry."
    exit 1
}

# PyTorch 2.5.x supports Python 3.9–3.12. 3.13 may fail.
$major, $minor = $pyVer.Split('.')
if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 9)) {
    Write-Err "Python $pyVer is too old. Please use Python 3.9–3.12."
    exit 1
}
if ([int]$major -eq 3 -and [int]$minor -ge 13) {
    Write-Warn "Python $pyVer may not have official PyTorch wheels yet. Prefer 3.9–3.12."
}

$pythonExe = $PythonPath

if ($CreateVenv) {
    Write-Section "Creating virtual environment (.venv)"
    if (-not (Test-Path ".venv")) {
        Exec "$PythonPath -m venv .venv"
    } else {
        Write-Host ".venv already exists; reusing it"
    }
    $pythonExe = Join-Path (Resolve-Path ".venv").Path "Scripts/python.exe"
}

Write-Section "Upgrading pip"
Exec "$pythonExe -m pip install --upgrade pip wheel setuptools"

Write-Section "Detecting NVIDIA GPU and CUDA"
$indexUrl = "https://download.pytorch.org/whl/cpu"
$cudaInfo = $null
try {
    $smi = & nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0 -and $smi) {
        $cudaLine = ($smi | Select-String -Pattern "CUDA Version:")
        if ($cudaLine) {
            $m = [regex]::Match($cudaLine.ToString(), "CUDA Version:\s*([0-9]+)\.([0-9]+)")
            if ($m.Success) {
                $cudaMajor = [int]$m.Groups[1].Value
                $cudaMinor = [int]$m.Groups[2].Value
                $cudaInfo = "$cudaMajor.$cudaMinor"
                Write-Host "CUDA Version detected: $cudaInfo"
                if ($cudaMajor -ge 12) {
                    $indexUrl = "https://download.pytorch.org/whl/cu121"
                } elseif ($cudaMajor -eq 11) {
                    $indexUrl = "https://download.pytorch.org/whl/cu118"
                } else {
                    Write-Warn "Unsupported/old CUDA $cudaInfo; using CPU wheels."
                    $indexUrl = "https://download.pytorch.org/whl/cpu"
                }
            }
        } else {
            Write-Warn "nvidia-smi found but CUDA version not parsed; defaulting to CPU wheels."
        }
    } else {
        Write-Host "No NVIDIA GPU detected; using CPU wheels."
    }
}
catch {
    Write-Host "nvidia-smi not available; using CPU wheels."
}

Write-Section "Installing PyTorch ($TorchVersion)"
$torchCmd = "$pythonExe -m pip install torch==$TorchVersion torchvision==$TorchVisionVersion torchaudio==$TorchAudioVersion --index-url $indexUrl"
Exec $torchCmd

Write-Section "Installing other dependencies"
if (-not (Test-Path $RequirementsPath)) {
    Write-Err "Requirements file not found at $RequirementsPath"
    exit 1
}
# Install all remaining deps; already-installed torch packages satisfy the >= constraints
Exec "$pythonExe -m pip install -r `"$RequirementsPath`""

Write-Section "Verifying installation"
try {
    $check = & $pythonExe - << 'PY'
import torch, torchvision, torchaudio
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
PY
    $check | ForEach-Object { Write-Host $_ }
}
catch {
    Write-Err "Verification failed. Try running: $torchCmd"
    exit 1
}

Write-Host "`nAll set! Activate the venv (if created) with:`n    .\\.venv\\Scripts\\Activate.ps1" -ForegroundColor Green
