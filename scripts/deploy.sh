#!/bin/bash
# Deployment script for LoRA Attention Analyzer package

set -e  # Exit on any error

echo "🚀 Deploying LoRA Attention Analyzer Package"
echo "============================================="

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
DEPLOY_TARGET="test"  # Default to test PyPI
FORCE_DEPLOY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            DEPLOY_TARGET="$2"
            shift 2
            ;;
        --force)
            FORCE_DEPLOY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--target test|prod] [--force]"
            echo "  --target test|prod: Deploy to test PyPI or production PyPI (default: test)"
            echo "  --force: Skip confirmation prompts"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate target
if [[ "$DEPLOY_TARGET" != "test" && "$DEPLOY_TARGET" != "prod" ]]; then
    print_error "Invalid target: $DEPLOY_TARGET. Must be 'test' or 'prod'"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
    print_error "setup.py or pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Check if dist/ directory exists and has files
if [ ! -d "dist/" ] || [ -z "$(ls -A dist/)" ]; then
    print_error "No built packages found in dist/ directory."
    print_status "Run './scripts/build_package.sh' first to build the package."
    exit 1
fi

# Display what will be deployed
print_status "Deployment Configuration:"
echo "========================="
echo "🎯 Target: $DEPLOY_TARGET PyPI"
echo "📦 Packages to deploy:"
ls -la dist/
echo ""

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    print_error "twine is not installed. Installing..."
    python -m pip install twine
fi

# Set repository based on target
if [ "$DEPLOY_TARGET" = "test" ]; then
    REPOSITORY="--repository testpypi"
    REPO_URL="https://test.pypi.org/"
    print_status "Deploying to Test PyPI..."
else
    REPOSITORY=""
    REPO_URL="https://pypi.org/"
    print_status "Deploying to Production PyPI..."
fi

# Final check of package
print_status "Performing final package check..."
python -m twine check dist/*
if [ $? -ne 0 ]; then
    print_error "Package check failed. Please fix the issues before deploying."
    exit 1
fi
print_success "Package check passed ✓"

# Get package version
PACKAGE_VERSION=$(python -c "import lora_attention_analyzer; print(lora_attention_analyzer.__version__)")
print_status "Package version: $PACKAGE_VERSION"

# Confirmation prompt (unless --force is used)
if [ "$FORCE_DEPLOY" = false ]; then
    echo ""
    print_warning "⚠️  You are about to deploy version $PACKAGE_VERSION to $DEPLOY_TARGET PyPI"
    echo ""
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Deployment cancelled."
        exit 0
    fi
fi

# Check for required environment variables/credentials
if [ "$DEPLOY_TARGET" = "prod" ]; then
    print_status "Checking production deployment credentials..."
    
    # Check if PyPI token is available
    if [ -z "$TWINE_PASSWORD" ] && [ -z "$TWINE_USERNAME" ]; then
        print_warning "No PyPI credentials found in environment variables."
        print_status "Please set TWINE_USERNAME and TWINE_PASSWORD, or configure ~/.pypirc"
        print_status "Alternatively, you'll be prompted for credentials during upload."
    fi
else
    print_status "Checking test PyPI credentials..."
    
    # For test PyPI, we might need different credentials
    if [ -z "$TEST_PYPI_TOKEN" ] && [ -z "$TWINE_PASSWORD" ]; then
        print_warning "No Test PyPI credentials found in environment variables."
        print_status "You'll be prompted for credentials during upload."
    fi
fi

# Perform the deployment
print_status "Starting deployment..."
echo "======================"

if [ "$DEPLOY_TARGET" = "test" ]; then
    python -m twine upload --repository testpypi dist/*
else
    python -m twine upload dist/*
fi

if [ $? -eq 0 ]; then
    print_success "🎉 Deployment successful!"
    echo ""
    echo "📋 Post-deployment steps:"
    echo "========================"
    
    if [ "$DEPLOY_TARGET" = "test" ]; then
        echo "🔗 Test PyPI page: https://test.pypi.org/project/lora-attention-analyzer/"
        echo "📦 Test installation: pip install --index-url https://test.pypi.org/simple/ lora-attention-analyzer"
        echo ""
        echo "🧪 To test the deployment:"
        echo "  1. Create a new virtual environment"
        echo "  2. pip install --index-url https://test.pypi.org/simple/ lora-attention-analyzer"
        echo "  3. Test import: python -c 'import lora_attention_analyzer; print(\"Success!\")'"
        echo ""
        echo "✅ Once testing is complete, deploy to production with:"
        echo "   ./scripts/deploy.sh --target prod"
    else
        echo "🔗 PyPI page: https://pypi.org/project/lora-attention-analyzer/"
        echo "📦 Installation: pip install lora-attention-analyzer"
        echo ""
        echo "📢 Announcement checklist:"
        echo "  ☐ Update README.md with new version"
        echo "  ☐ Create GitHub release with changelog"
        echo "  ☐ Update documentation"
        echo "  ☐ Announce on relevant forums/communities"
        echo "  ☐ Update examples if needed"
    fi
    
    echo ""
    echo "📊 Package information:"
    echo "  Version: $PACKAGE_VERSION"
    echo "  Repository: $REPO_URL"
    echo "  Upload time: $(date)"
    
else
    print_error "❌ Deployment failed!"
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "  1. Check your credentials"
    echo "  2. Verify package version is not already published"
    echo "  3. Ensure package passes all checks"
    echo "  4. Check network connectivity"
    echo ""
    echo "📝 Common solutions:"
    echo "  - Version conflicts: Update version in setup.py/pyproject.toml"
    echo "  - Credential issues: Check ~/.pypirc or environment variables"
    echo "  - Package issues: Run './scripts/build_package.sh' again"
    exit 1
fi