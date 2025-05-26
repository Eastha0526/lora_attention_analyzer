# Installation Guide

This guide provides detailed instructions for installing and setting up the LoRA Attention Analyzer package.

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 18.04+), Windows 10+, or macOS 10.15+
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **VRAM**: 8GB minimum, 16GB+ recommended for SDXL models
- **RAM**: 16GB minimum, 32GB+ recommended

### CUDA Setup (Linux/Windows)
For GPU acceleration, ensure CUDA is properly installed:

```bash
# Check CUDA version
nvidia-smi

# Install CUDA toolkit if needed (example for Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

## Installation Methods

### Method 1: Install from PyPI (Recommended when published)

```bash
# Basic installation
pip install lora-attention-analyzer

# With optional dependencies
pip install "lora-attention-analyzer[optional]"

# Development installation
pip install "lora-attention-analyzer[dev]"
```

### Method 2: Install from Source

#### Clone and Install
```bash
# Clone the repository
git clone https://github.com/yourusername/lora-attention-analyzer.git
cd lora-attention-analyzer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[optional,dev]"
```

#### Alternative: Direct pip install from Git
```bash
pip install git+https://github.com/yourusername/lora-attention-analyzer.git
```

### Method 3: Manual Installation

```bash
# Download the source code
wget https://github.com/yourusername/lora-attention-analyzer/archive/main.zip
unzip main.zip
cd lora-attention-analyzer-main

# Install dependencies
pip install -r requirements.txt

# Install the package
python setup.py install
```

## Dependency Installation

### Core Dependencies
These are automatically installed with the package:

```bash
pip install torch>=1.13.0
pip install torchvision>=0.14.0
pip install diffusers>=0.21.0
pip install transformers>=4.25.0
pip install daam
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install numpy>=1.21.0
pip install Pillow>=8.0.0
pip install safetensors>=0.3.0
pip install xformers>=0.0.16
pip install fire>=0.4.0
```

### Optional Dependencies
For enhanced functionality:

```bash
# Advanced image processing
pip install scikit-image>=0.19.0
pip install scipy>=1.8.0
```

### Development Dependencies
For contributing to the project:

```bash
pip install pytest>=6.0
pip install pytest-cov
pip install black
pip install flake8
pip install isort
```

## Platform-Specific Instructions

### Ubuntu/Debian
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip if not available
sudo apt install python3 python3-pip python3-venv -y

# Install system dependencies for image processing
sudo apt install libjpeg-dev zlib1g-dev libpng-dev -y

# Install git if needed
sudo apt install git -y

# Follow standard installation
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install lora-attention-analyzer
```

### CentOS/RHEL/Fedora
```bash
# Install Python and dependencies
sudo dnf install python3 python3-pip python3-devel git -y  # Fedora
# OR
sudo yum install python3 python3-pip python3-devel git -y  # CentOS/RHEL

# Install image processing libraries
sudo dnf install libjpeg-turbo-devel zlib-devel libpng-devel -y

# Follow standard installation
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install lora-attention-analyzer
```

### Windows
```powershell
# Install Python from python.org if not available
# Make sure to add Python to PATH during installation

# Install git from git-scm.com if needed

# Open Command Prompt or PowerShell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install lora-attention-analyzer
```

### macOS
```bash
# Install Homebrew if not available
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and git
brew install python git

# Follow standard installation
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install lora-attention-analyzer
```

## Docker Installation

### Using Pre-built Image (when available)
```bash
docker pull yourusername/lora-attention-analyzer:latest
docker run -it --gpus all yourusername/lora-attention-analyzer:latest
```

### Build from Source
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy and install package
COPY . .
RUN pip3 install -e .

# Set entrypoint
ENTRYPOINT ["lora-attention-analyzer"]
```

```bash
# Build and run
docker build -t lora-attention-analyzer .
docker run -it --gpus all -v /path/to/models:/models lora-attention-analyzer
```

## Conda Installation

### Create Environment
```bash
# Create conda environment
conda create -n lora-attention python=3.10 -y
conda activate lora-attention

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install the package
pip install lora-attention-analyzer
```

### Environment File
Create `environment.yml`:
```yaml
name: lora-attention
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - pytorch-cuda=11.8
  - pip
  - pip:
    - lora-attention-analyzer
```

```bash
conda env create -f environment.yml
conda activate lora-attention
```

## Verification

### Test Installation
```bash
# Test import
python -c "import lora_attention_analyzer; print('✅ Installation successful!')"

# Test CLI
lora-attention-analyzer --help

# Check version
python -c "import lora_attention_analyzer; print(f'Version: {lora_attention_analyzer.__version__}')"
```

### Run Example
```bash
# Analyze a LoRA file (replace with your file path)
lora-attention-analyzer analyze --lora_file /path/to/your/lora.safetensors
```

### GPU Test
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solution: Use CPU or reduce batch size
export CUDA_VISIBLE_DEVICES=""  # Force CPU
# Or install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Package Not Found
```bash
# Make sure pip is up to date
pip install --upgrade pip

# Clear pip cache
pip cache purge

# Try installing with verbose output
pip install -v lora-attention-analyzer
```

#### 3. Permission Errors (Linux/macOS)
```bash
# Use user installation
pip install --user lora-attention-analyzer

# Or fix permissions
sudo chown -R $USER:$USER ~/.local/
```

#### 4. Import Errors
```bash
# Check installed packages
pip list | grep lora

# Reinstall with dependencies
pip uninstall lora-attention-analyzer
pip install lora-attention-analyzer --no-cache-dir
```

#### 5. xformers Issues
```bash
# Install specific xformers version
pip install xformers==0.0.16

# Or install from source
pip install git+https://github.com/facebookresearch/xformers.git
```

### Getting Help

1. **Check Requirements**: Ensure all system requirements are met
2. **Update Packages**: Run `pip install --upgrade lora-attention-analyzer`
3. **Check Issues**: Visit [GitHub Issues](https://github.com/yourusername/lora-attention-analyzer/issues)
4. **Verbose Installation**: Use `pip install -v` for detailed output
5. **Clean Installation**: Remove and reinstall in a fresh environment

### Performance Optimization

#### For Better Performance
```bash
# Install optimized PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Enable memory efficient attention
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use mixed precision
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For Apple Silicon
```

#### Memory Optimization
```python
# In your scripts
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

## Next Steps

After successful installation:

1. **Read the Documentation**: Check the main README.md for usage examples
2. **Run Examples**: Try the examples in the `examples/` directory
3. **Prepare Your Models**: Download and organize your SDXL models and LoRA files
4. **Start Analyzing**: Begin with simple comparisons using the CLI

For detailed usage instructions, see the main [README.md](README.md) file.