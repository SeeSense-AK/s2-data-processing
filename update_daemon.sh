#!/bin/bash

echo "Updating pipeline daemon..."

# Unload current daemon
sudo launchctl unload /Library/LaunchDaemons/com.seesense.pipeline.daemon.plist

# Copy updated plist
sudo cp com.seesense.pipeline.daemon.plist /Library/LaunchDaemons/

# Set ownership
sudo chown root:wheel /Library/LaunchDaemons/com.seesense.pipeline.daemon.plist

# Load updated daemon
sudo launchctl load /Library/LaunchDaemons/com.seesense.pipeline.daemon.plist

echo "✅ Daemon updated!"
echo "To test: sudo launchctl start com.seesense.pipeline.daemon"