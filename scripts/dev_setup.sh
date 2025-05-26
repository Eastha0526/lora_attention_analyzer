#!/bin/bash
# Development environment setup script for LoRA Attention Analyzer

set -e  # Exit on any error

echo "🛠️  Setting up LoRA Attention Analyzer Development Environment"
echo "============================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
SKIP_VENV=false
SKIP_TESTS=false
INSTALL_GPU=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --cpu-only)
            INSTALL_GPU=false
            shift
            ;;
        --help)
            echo "Usage: $0 [--skip-venv] [--skip-tests] [--cpu-only]"
            echo "  --skip-venv: Skip virtual environment creation"
            echo "  --skip-tests: Skip running tests after setup"
            echo "  --cpu-only: Install CPU-only version of PyTorch"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
    print_error "setup.py or pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Check Python version
print_status "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi
print_success "Python version: $python_version ✓"

# Check if git is available
if ! command -v git &> /dev/null; then
    print_warning "Git not found. Some development features may not work."
else
    print_success "Git available ✓"
fi

# Create virtual environment
if [ "$SKIP_VENV" = false ]; then
    print_status "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old environment..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    print_success "Virtual environment created ✓"
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated ✓"
else
    print_warning "Skipping virtual environment creation (--skip-venv specified)"
fi

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip
print_success "pip upgraded ✓"

# Install build tools
print_status "Installing build tools..."
python -m pip install --upgrade wheel setuptools build twine
print_success "Build tools installed ✓"

# Install PyTorch (with or without CUDA)
print_status "Installing PyTorch..."
if [ "$INSTALL_GPU" = true ]; then
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected, installing CUDA version..."
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        print_warning "No NVIDIA GPU detected, installing CPU version..."
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
else
    print_status "Installing CPU-only PyTorch..."
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
print_success "PyTorch installed ✓"

# Install package in development mode
print_status "Installing package in development mode..."
python -m pip install -e ".[dev,optional]"
print_success "Package installed in development mode ✓"

# Install pre-commit hooks if available
if command -v git &> /dev/null && [ -d ".git" ]; then
    print_status "Setting up pre-commit hooks..."
    
    # Create pre-commit config if it doesn't exist
    if [ ! -f ".pre-commit-config.yaml" ]; then
        cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
EOF
        print_status "Created .pre-commit-config.yaml"
    fi
    
    python -m pip install pre-commit
    pre-commit install
    print_success "Pre-commit hooks installed ✓"
else
    print_warning "Skipping pre-commit hooks (not a git repository or git not available)"
fi

# Create directories if they don't exist
print_status "Creating project directories..."
mkdir -p logs
mkdir -p data
mkdir -p models
mkdir -p output
print_success "Project directories created ✓"

# Run tests to verify installation
if [ "$SKIP_TESTS" = false ]; then
    print_status "Running tests to verify installation..."
    
    if python -m pytest tests/ -v --tb=short; then
        print_success "All tests passed ✓"
    else
        print_warning "Some tests failed. This might be expected if you don't have model files."
        print_status "You can run tests later with: python -m pytest tests/"
    fi
else
    print_warning "Skipping tests (--skip-tests specified)"
fi

# Verify installation
print_status "Verifying installation..."

# Test import
if python -c "import lora_attention_analyzer; print('Import successful')" > /dev/null 2>&1; then
    print_success "Package import successful ✓"
else
    print_error "Package import failed ✗"
    exit 1
fi

# Test CLI
if lora-attention-analyzer --help > /dev/null 2>&1; then
    print_success "CLI tool available ✓"
else
    print_error "CLI tool not available ✗"
    exit 1
fi

# Check GPU availability
print_status "Checking GPU availability..."
gpu_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$gpu_available" = "True" ]; then
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    print_success "GPU available: $gpu_count GPU(s) detected ✓"
else
    print_warning "No GPU available (using CPU mode)"
fi

# Display setup summary
echo ""
print_success "🎉 Development environment setup completed!"
echo ""
echo "Setup Summary:"
echo "================="
echo "Python version: $python_version"
echo "Virtual environment: $(if [ "$SKIP_VENV" = false ]; then echo "Created"; else echo "Skipped"; fi)"
echo "PyTorch: $(if [ "$INSTALL_GPU" = true ]; then echo "GPU version"; else echo "CPU version"; fi)"
echo "GPU support: $(if [ "$gpu_available" = "True" ]; then echo "Available"; else echo "Not available"; fi)"
echo ""
echo "Project structure:"
echo "  lora_attention_analyzer/  - Main package"
echo "  tests/                   - Test suite"
echo "  examples/                - Usage examples"
echo "  logs/                    - Log files"
echo "  data/                    - Data files"
echo "  models/                  - Model files"
echo "  output/                  - Analysis results"
echo ""
echo "Development commands:"
echo "====================="
echo "Run tests:           python -m pytest tests/"
echo "Format code:         python -m black lora_attention_analyzer/"
echo "Sort imports:        python -m isort lora_attention_analyzer/"
echo "Lint code:           python -m flake8 lora_attention_analyzer/"
echo "Build package:       ./scripts/build_package.sh"
echo "Deploy package:      ./scripts/deploy.sh"
echo ""
echo "Next steps:"
echo "==========="
echo "1. Download some SDXL models and LoRA files"
echo "2. Place them in the models/ directory"
echo "3. Run examples: python examples/basic_usage.py"
echo "4. Start analyzing: lora-attention-analyzer --help"
echo ""

# Activation reminder
if [ "$SKIP_VENV" = false ]; then
    echo "To activate the environment in the future:"
    echo "   source venv/bin/activate"
    echo ""
fi

print_success "Happy developing!"