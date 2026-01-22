#!/bin/bash
# Install Docker Health Check as Launch Daemon for headless operation

set -e

echo "🔧 Installing Docker Health Check Launch Daemon..."

# Copy plist to system location
echo "📁 Installing plist to /Library/LaunchDaemons/..."
sudo cp /Users/abhishekkumbhar/Documents/s2-data-processing/dockerhealthchecks/com.seesense.docker.health.daemon.plist /Library/LaunchDaemons/

# Set proper permissions
echo "🔐 Setting permissions..."
sudo chown root:wheel /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist
sudo chmod 644 /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist

# Validate plist
echo "✅ Validating plist syntax..."
plutil -lint /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist

# Create log directory if it doesn't exist
echo "📝 Ensuring log directory exists..."
sudo mkdir -p /var/log/seesense
sudo chown abhishekkumbhar:staff /var/log/seesense

# Load the daemon
echo "🚀 Loading Launch Daemon..."
sudo launchctl bootstrap system /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist

# Enable the daemon
echo "⚡ Enabling Launch Daemon..."
sudo launchctl enable system/com.seesense.docker.health.daemon

# Check status
echo "📊 Checking daemon status..."
sudo launchctl print system/com.seesense.docker.health.daemon

echo ""
echo "✅ Launch Daemon installed successfully!"
echo ""
echo "📋 Summary:"
echo "  • Service: com.seesense.docker.health.daemon"
echo "  • Schedule: Daily at 2:30 AM"
echo "  • Runs as: abhishekkumbhar (with Docker access)"
echo "  • Works on: Headless systems"
echo "  • Logs: /var/log/seesense/"
echo ""
echo "🧪 To test manually:"
echo "  sudo launchctl kickstart system/com.seesense.docker.health.daemon"
echo ""
echo "🔍 To check status:"
echo "  sudo launchctl print system/com.seesense.docker.health.daemon"
echo ""