#!/bin/bash
# Test Docker Health Check Launch Daemon

echo "🧪 Testing Launch Daemon Installation..."
echo ""

# Check if plist exists
if [[ -f "/Library/LaunchDaemons/com.seesense.docker.health.daemon.plist" ]]; then
    echo "✅ Launch Daemon plist found"
else
    echo "❌ Launch Daemon plist not found - run install_daemon.sh first"
    exit 1
fi

# Check if daemon is loaded
echo "📊 Checking daemon status..."
if sudo launchctl print system/com.seesense.docker.health.daemon >/dev/null 2>&1; then
    echo "✅ Launch Daemon is loaded"
    echo ""
    echo "📋 Daemon Details:"
    sudo launchctl print system/com.seesense.docker.health.daemon | head -20
else
    echo "❌ Launch Daemon is not loaded - run install_daemon.sh first"
    exit 1
fi

echo ""
echo "🔥 Manual test (this will run the health check now):"
echo "sudo launchctl kickstart system/com.seesense.docker.health.daemon"
echo ""
echo "📝 Check logs:"
echo "tail -f /var/log/seesense/docker_health_daemon.out"
echo "tail -f /var/log/seesense/docker_health_check.log"