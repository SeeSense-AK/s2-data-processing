#!/bin/bash

# S2 Data Processing Pipeline Setup Script
# Creates directory structure and validates environment

set -e  # Exit on any error

echo "🚀 Setting up S2 Data Processing Pipeline"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"

# Create main directory structure
echo -e "\n📁 Creating directory structure..."

directories=(
    "data"
    "data/downloadedfiles"
    "data/combinedfile" 
    "data/preprocessed"
    "data/processed"
    "logs"
    "config"
    "scripts"
    "scripts/utils"
    "tests"
)

for dir in "${directories[@]}"; do
    mkdir -p "$PROJECT_ROOT/$dir"
    print_status "Created $dir/"
done

# Create region subdirectories
echo -e "\n🌍 Creating region subdirectories..."

regions=("england" "finland" "ireland" "sydney" "wales")

for region in "${regions[@]}"; do
    mkdir -p "$PROJECT_ROOT/data/preprocessed/$region"
    mkdir -p "$PROJECT_ROOT/data/processed/$region"
    print_status "Created region directories for $region"
done

# Create .gitkeep files to preserve empty directories
echo -e "\n📝 Creating .gitkeep files..."

gitkeep_dirs=(
    "data/downloadedfiles"
    "data/combinedfile"
    "logs"
)

for dir in "${gitkeep_dirs[@]}"; do
    touch "$PROJECT_ROOT/$dir/.gitkeep"
    print_status "Created $dir/.gitkeep"
done

# Check for required files
echo -e "\n🔍 Checking configuration files..."

config_files=(
    "config/pipeline_config.json:Pipeline configuration"
    "config/device_mapping.json:Device mapping configuration"
)

for config_info in "${config_files[@]}"; do
    IFS=':' read -r file desc <<< "$config_info"
    if [ -f "$PROJECT_ROOT/$file" ]; then
        print_status "$desc exists"
    else
        print_warning "$desc missing: $file"
        echo "   Please create this file before running the pipeline"
    fi
done

# Check for AWS configuration
if [ -f "$PROJECT_ROOT/config/aws_config.json" ]; then
    print_status "AWS configuration exists"
else
    print_warning "AWS configuration missing: config/aws_config.json"
    if [ -f "$PROJECT_ROOT/config/aws_config_template.json" ]; then
        echo "   Template available at config/aws_config_template.json"
        echo "   Copy and modify: cp config/aws_config_template.json config/aws_config.json"
    fi
fi

# Check Python environment
echo -e "\n🐍 Checking Python environment..."

if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    print_status "Python3 found: $python_version"
else
    print_error "Python3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Check for conda
if command -v conda &> /dev/null; then
    print_status "Conda found"
    
    # Check if environment exists
    if conda env list | grep -q "s2-data-processing"; then
        print_status "Conda environment 's2-data-processing' exists"
    else
        print_warning "Conda environment 's2-data-processing' not found"
        echo "   Create it with: conda env create -f environment.yml"
    fi
else
    print_warning "Conda not found. Using system Python."
fi

# Check Docker and OSRM containers
echo -e "\n🐳 Checking Docker and OSRM containers..."

if command -v docker &> /dev/null; then
    print_status "Docker found"
    
    # Check if Docker is running
    if docker info &> /dev/null; then
        print_status "Docker is running"
        
        # Check OSRM containers
        osrm_containers=("osrm-england:5005" "osrm-finland:5001" "osrm-ireland:5002" "osrm-sydney:5003" "osrm-wales:5004")
        
        for container_info in "${osrm_containers[@]}"; do
            IFS=':' read -r container port <<< "$container_info"
            
            if docker ps | grep -q "$container"; then
                print_status "$container is running on port $port"
            else
                print_warning "$container is not running"
                echo "   Start with: docker start $container"
            fi
        done
    else
        print_warning "Docker is not running"
        echo "   Start Docker and then check OSRM containers"
    fi
else
    print_error "Docker not found. Please install Docker to run OSRM servers."
fi

# Check required Python packages
echo -e "\n📦 Checking Python packages..."

required_packages=("pandas" "numpy" "requests" "tqdm" "boto3" "geopy" "polyline" "folium")

for package in "${required_packages[@]}"; do
    if python3 -c "import $package" &> /dev/null; then
        print_status "$package is installed"
    else
        print_warning "$package is not installed"
        echo "   Install with: pip install $package"
    fi
done

# Set up example cron job
echo -e "\n⏰ Setting up scheduled execution..."

cron_example="# S2 Data Pipeline - Run daily at 3:30 AM
30 3 * * * cd $PROJECT_ROOT && $PROJECT_ROOT/scripts/main_pipeline.py --automated"

echo "Example cron job entry:"
echo "$cron_example"
echo ""
echo "To add to crontab:"
echo "1. Run: crontab -e"
echo "2. Add the above lines"
echo "3. Save and exit"

# Create sample AWS config template if it doesn't exist
if [ ! -f "$PROJECT_ROOT/config/aws_config_template.json" ]; then
    echo -e "\n📄 Creating AWS config template..."
    
    cat > "$PROJECT_ROOT/config/aws_config_template.json" << 'EOF'
{
  "aws_access_key_id": "YOUR_ACCESS_KEY_ID",
  "aws_secret_access_key": "YOUR_SECRET_ACCESS_KEY",
  "region": "eu-west-1"
}
EOF
    
    print_status "Created config/aws_config_template.json"
fi

# Display summary
echo -e "\n📋 Setup Summary"
echo "================"
echo "✅ Directory structure created"
echo "✅ Configuration templates available"
echo ""

print_info "Next steps:"
echo "1. Configure AWS credentials in config/aws_config.json"
echo "2. Update device mapping in config/device_mapping.json"
echo "3. Ensure OSRM Docker containers are running"
echo "4. Test the pipeline with: python scripts/tests/test_step3.py"
echo "5. Run the complete pipeline: python scripts/main_pipeline.py"
echo ""

print_info "For automated daily runs:"
echo "1. Add the cron job shown above"
echo "2. Ensure the conda environment is activated in the cron job if needed"
echo ""

print_status "Setup completed successfully!"

exit 0