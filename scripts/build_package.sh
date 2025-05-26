#!/bin/bash
# Build script for LoRA Attention Analyzer package

set -e  # Exit on any error

echo "🏗️  Building LoRA Attention Analyzer Package"
echo "=============================================="

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

# Check if we're in the right directory
if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
    print_error "setup.py or pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Check Python version
print_status "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.8"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi
print_success "Python version: $python_version ✓"

# Clean previous builds
print_status "Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
print_success "Clean completed ✓"

# Update build tools
print_status "Updating build tools..."
python -m pip install --upgrade pip
python -m pip install --upgrade build twine wheel setuptools
print_success "Build tools updated ✓"

# Install package in development mode for testing
print_status "Installing package in development mode..."
python -m pip install -e ".[dev]"
print_success "Development installation completed ✓"

# Run tests
print_status "Running tests..."
if command -v pytest &> /dev/null; then
    python -m pytest tests/ -v --tb=short
    if [ $? -ne 0 ]; then
        print_warning "Some tests failed, but continuing with build..."
    else
        print_success "All tests passed ✓"
    fi
else
    print_warning "pytest not found, skipping tests"
fi

# Run linting (optional)
print_status "Running code quality checks..."
if command -v black &> /dev/null; then
    print_status "Running black formatter..."
    python -m black --check lora_attention_analyzer/ || print_warning "Code formatting issues found"
fi

if command -v flake8 &> /dev/null; then
    print_status "Running flake8 linter..."
    python -m flake8 lora_attention_analyzer/ --max-line-length=88 --extend-ignore=E203,W503 || print_warning "Linting issues found"
fi

if command -v isort &> /dev/null; then
    print_status "Running isort import sorter..."
    python -m isort --check-only lora_attention_analyzer/ || print_warning "Import sorting issues found"
fi

# Build package
print_status "Building package..."
python -m build

if [ $? -eq 0 ]; then
    print_success "Package built successfully ✓"
else
    print_error "Package build failed"
    exit 1
fi

# Check built package
print_status "Checking built package..."
python -m twine check dist/*

if [ $? -eq 0 ]; then
    print_success "Package check passed ✓"
else
    print_error "Package check failed"
    exit 1
fi

# Display build information
print_status "Build Summary:"
echo "=============="
echo "📦 Built packages:"
ls -la dist/
echo ""
echo "📊 Package sizes:"
du -h dist/*
echo ""

# Check package contents
print_status "Package contents preview:"
if command -v tar &> /dev/null; then
    echo "🗂️  Source distribution contents:"
    tar -tzf dist/*.tar.gz | head -20
    if [ $(tar -tzf dist/*.tar.gz | wc -l) -gt 20 ]; then
        echo "... (showing first 20 files)"
    fi
fi

echo ""
print_success "Build completed successfully! 🎉"
echo ""
echo "📋 Next steps:"
echo "  1. Test installation: pip install dist/*.whl"
echo "  2. Upload to test PyPI: twine upload --repository testpypi dist/*"
echo "  3. Upload to PyPI: twine upload dist/*"
echo ""
echo "🔧 Useful commands:"
echo "  - Install locally: pip install dist/*.whl"
echo "  - Upload to TestPyPI: python -m twine upload --repository testpypi dist/*"
echo "  - Upload to PyPI: python -m twine upload dist/*"