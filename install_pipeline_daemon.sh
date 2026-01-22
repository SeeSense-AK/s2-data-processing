#!/bin/bash

echo "Installing S2 Pipeline Daemon for daily 3:30am execution..."

# Copy the plist to LaunchDaemons
sudo cp com.seesense.pipeline.daemon.plist /Library/LaunchDaemons/

# Set proper ownership
sudo chown root:wheel /Library/LaunchDaemons/com.seesense.pipeline.daemon.plist

# Load the daemon
sudo launchctl load /Library/LaunchDaemons/com.seesense.pipeline.daemon.plist

echo "✅ Daemon installed successfully!"
echo "📅 Your pipeline will run daily at 3:30 AM"
echo "📝 Logs will be in: logs/daemon.out and logs/daemon.err"
echo ""
echo "To check if it's loaded: sudo launchctl list | grep com.seesense"
echo "To uninstall: sudo launchctl unload /Library/LaunchDaemons/com.seesense.pipeline.daemon.plist"