[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [string]$Python = ".venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path $Python)) {
    throw "Python environment missing: $Python. Run py -3.12 -m venv .venv first (tested Python 3.12.14)."
}
function Run-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}
Run-Python -Arguments @("-c", "import sys; print('Python', sys.version.split()[0]); assert sys.version_info[:2] == (3, 12), 'Use Python 3.12 for the demonstrated pinned environment (py -3.12 -m venv .venv).'")
Run-Python -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Run-Python -Arguments @("-m", "pip", "install", "-r", "requirements.txt")
$WheelIndex = "https://download.pytorch.org/whl/cu128"
$TorchBuild = "2.7.1+cu128"
$VisionBuild = "0.22.1+cu128"
if ($CpuOnly) {
    $WheelIndex = "https://download.pytorch.org/whl/cpu"
    $TorchBuild = "2.7.1+cpu"
    $VisionBuild = "0.22.1+cpu"
}
# Exact local build suffixes replace an existing CPU build when installing CUDA.
# CUDA 12.8 PyTorch wheels support Blackwell GPUs. No system CUDA toolkit is
# required for these wheels, but a compatible NVIDIA driver must be installed.
Run-Python -Arguments @("-m", "pip", "install", "torch==$TorchBuild", "torchvision==$VisionBuild", "--index-url", $WheelIndex)
Run-Python -Arguments @("-m", "pip", "install", "-r", "requirements-neural.txt")
Run-Python -Arguments @("-m", "pip", "check")
Run-Python -Arguments @("-c", "import torch; from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel; print('PyTorch', torch.__version__, 'CUDA build', torch.version.cuda, 'available', torch.cuda.is_available()); print('Actual compute device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); x=torch.ones((16,16),device='cuda' if torch.cuda.is_available() else 'cpu'); print('Executed matrix multiplication:',float((x@x).sum().item()))")
Write-Host "Neural packages installed. A GPU was used only if the actual compute probe above reported CUDA."
Write-Host "Training also requires genuine ImageNet/ImageNette photographs for the penalty dataset."
Write-Host "Set models.efficientad.imagenet_dir or enable models.efficientad.download_imagenette."
Write-Host "Teacher weights download with a published Anomalib SHA256 check; offline cache is reused."
