#!/bin/bash
# Installation script for Docker Health Check Launch Agent
# Run this script to install the health check system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_NAME="docker_health_check.sh"
PLIST_NAME="com.seesense.docker.health.check.plist"
SCRIPT_DIR="/usr/local/bin/seesense"
LOG_DIR="/var/log/seesense"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"

# Print colored output
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

# Check if running as root for some operations
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        print_error "Don't run this script as root. It will request sudo when needed."
        exit 1
    fi
}

# Validate Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check OSRM containers exist
check_containers() {
    print_status "Checking OSRM containers..."
    
    local required_containers=("osrm-england" "osrm-finland" "osrm-ireland" "osrm-sydney" "osrm-wales")
    local missing_containers=()
    
    for container in "${required_containers[@]}"; do
        if docker ps -a --format "table {{.Names}}" | grep -q "^${container}$"; then
            print_success "Found container: $container"
        else
            missing_containers+=("$container")
            print_warning "Missing container: $container"
        fi
    done
    
    if [[ ${#missing_containers[@]} -gt 0 ]]; then
        print_error "Missing containers: ${missing_containers[*]}"
        print_error "Please create these containers first."
        exit 1
    fi
    
    print_success "All required OSRM containers found"
}

# Create directories
create_directories() {
    print_status "Creating directories..."
    
    # Create script directory (requires sudo)
    if [[ ! -d "$SCRIPT_DIR" ]]; then
        sudo mkdir -p "$SCRIPT_DIR"
        print_success "Created script directory: $SCRIPT_DIR"
    fi
    
    # Create log directory (requires sudo)
    if [[ ! -d "$LOG_DIR" ]]; then
        sudo mkdir -p "$LOG_DIR"
        sudo chown "$USER:staff" "$LOG_DIR"
        print_success "Created log directory: $LOG_DIR"
    fi
    
    # Create LaunchAgent directory
    mkdir -p "$LAUNCHD_DIR"
    print_success "Ensured LaunchAgent directory exists"
}

# Install health check script
install_script() {
    print_status "Installing health check script..."
    
    # Check if script exists in current directory
    if [[ ! -f "$SCRIPT_NAME" ]]; then
        print_error "Health check script not found in current directory"
        print_error "Please ensure $SCRIPT_NAME exists before running this installer"
        exit 1
    fi
    
    # Copy script to system location
    sudo cp "$SCRIPT_NAME" "$SCRIPT_DIR/"
    sudo chmod +x "$SCRIPT_DIR/$SCRIPT_NAME"
    sudo chown root:wheel "$SCRIPT_DIR/$SCRIPT_NAME"
    
    print_success "Installed script to $SCRIPT_DIR/$SCRIPT_NAME"
}

# Install launch agent plist
install_plist() {
    print_status "Installing Launch Agent..."
    
    # Check if plist exists
    if [[ ! -f "$PLIST_NAME" ]]; then
        print_error "Launch Agent plist not found in current directory"
        print_error "Please ensure $PLIST_NAME exists before running this installer"
        exit 1
    fi
    
    # Copy plist to LaunchAgents directory
    cp "$PLIST_NAME" "$LAUNCHD_DIR/"
    
    print_success "Installed plist to $LAUNCHD_DIR/$PLIST_NAME"
}

# Load and start the launch agent
load_launch_agent() {
    print_status "Loading Launch Agent..."
    
    # Unload if already loaded (ignore errors)
    launchctl unload "$LAUNCHD_DIR/$PLIST_NAME" 2>/dev/null || true
    
    # Load the new agent
    if launchctl load "$LAUNCHD_DIR/$PLIST_NAME"; then
        print_success "Launch Agent loaded successfully"
    else
        print_error "Failed to load Launch Agent"
        exit 1
    fi
    
    # Verify it's loaded
    if launchctl list | grep -q "com.seesense.docker.health.check"; then
        print_success "Launch Agent is active and scheduled"
    else
        print_warning "Launch Agent loaded but may not be active"
    fi
}

# Test the installation
test_installation() {
    print_status "Testing installation..."
    
    # Test script execution
    print_status "Testing health check script..."
    if sudo "$SCRIPT_DIR/$SCRIPT_NAME" test; then
        print_success "Health check script test passed"
    else
        print_error "Health check script test failed"
    fi
    
    # Show schedule
    print_status "Verifying schedule..."
    if launchctl list | grep -q "com.seesense.docker.health.check"; then
        print_success "Launch Agent is scheduled to run daily at 2:30 AM"
    else
        print_error "Launch Agent is not properly scheduled"
    fi
}

# Slack webhook configuration
configure_slack() {
    print_status "Configuring Slack notifications..."
    
    echo
    echo "To enable Slack notifications:"
    echo "1. Create a Slack webhook URL in your workspace"
    echo "2. Edit the health check script: sudo nano $SCRIPT_DIR/$SCRIPT_NAME"
    echo "3. Replace 'YOUR_SLACK_WEBHOOK_URL_HERE' with your actual webhook URL"
    echo
    print_warning "Slack notifications are optional but recommended for monitoring"
}

# Show status and next steps
show_status() {
    echo
    echo "=========================================="
    print_success "Docker Health Check Installation Complete!"
    echo "=========================================="
    echo
    echo "📋 What was installed:"
    echo "  • Health check script: $SCRIPT_DIR/$SCRIPT_NAME"
    echo "  • Launch Agent: $LAUNCHD_DIR/$PLIST_NAME"
    echo "  • Log directory: $LOG_DIR"
    echo
    echo "🕐 Schedule:"
    echo "  • Health check runs daily at 2:30 AM"
    echo "  • Your pipeline runs daily at 3:30 AM"
    echo "  • 1-hour buffer ensures containers are ready"
    echo
    echo "📋 Manual commands:"
    echo "  • Test health check: sudo $SCRIPT_DIR/$SCRIPT_NAME test"
    echo "  • Start all containers: sudo $SCRIPT_DIR/$SCRIPT_NAME start-all"
    echo "  • View logs: tail -f $LOG_DIR/docker_health_check.log"
    echo "  • Check agent status: launchctl list | grep docker.health"
    echo
    echo "🔧 Troubleshooting:"
    echo "  • Unload agent: launchctl unload $LAUNCHD_DIR/$PLIST_NAME"
    echo "  • Reload agent: launchctl load $LAUNCHD_DIR/$PLIST_NAME"
    echo "  • Edit script: sudo nano $SCRIPT_DIR/$SCRIPT_NAME"
    echo
}

# Main installation process
main() {
    echo "=========================================="
    echo "🐳 Docker OSRM Health Check Installer"
    echo "=========================================="
    echo
    
    check_permissions
    check_docker
    check_containers
    create_directories
    install_script
    install_plist
    load_launch_agent
    test_installation
    configure_slack
    show_status
    
    print_success "Installation completed successfully!"
    print_status "The health check will run automatically at 2:30 AM daily"
}

# Run main function
main "$@"