# S2 Data Processing Pipeline - Automation Documentation

> **Complete guide for automated S2 data processing on headless Mac Mini**
> Created: September 25, 2025
> System: macOS running 24/7 headless operation

---

## 📋 Overview

This document describes the complete automation setup for the S2 data processing pipeline, designed to run unattended on a headless Mac Mini. The system includes dual-layer monitoring with Docker health checks and comprehensive Slack notifications.

### **Daily Automation Schedule**
- **2:30 AM**: Docker health check (Launch Daemon)
- **3:30 AM**: S2 data pipeline execution (Cron job)

---

## 🏗️ Architecture

### **System Components**

1. **Launch Daemon** (`com.seesense.docker.health.daemon`)
   - Monitors OSRM Docker containers
   - Runs at system level (headless compatible)
   - Automatically restarts failed containers

2. **Cron Job** (User-level)
   - Executes main S2 pipeline
   - Processes previous day's data
   - Includes built-in container checks as backup

3. **Slack Integration**
   - Real-time notifications for both systems
   - Success/failure reporting with logs
   - Webhook-based messaging

---

## 🔧 Installation & Setup

### **Prerequisites**
- macOS system with Docker Desktop
- Conda environment: `s2-data-processing`
- OSRM containers running on ports 5001-5005
- Slack workspace with incoming webhook

### **Required OSRM Containers**
```bash
osrm-finland   (port 5001)
osrm-ireland   (port 5002)
osrm-sydney    (port 5003)
osrm-wales     (port 5004)
osrm-england   (port 5005)
```

---

## ⚙️ Configuration Files

### **1. Launch Daemon Configuration**
**Location**: `/Library/LaunchDaemons/com.seesense.docker.health.daemon.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.seesense.docker.health.daemon</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/seesense/docker_health_check.sh</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
        <key>SLACK_WEBHOOK_URL</key>
        <string>YOUR_WEBHOOK_URL_HERE</string>
    </dict>

    <key>UserName</key>
    <string>abhishekkumbhar</string>
    <key>SessionCreate</key>
    <false/>
</dict>
</plist>
```

### **2. Cron Configuration**
```bash
# Environment variable for Slack notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Daily pipeline execution at 3:30 AM
30 3 * * * /Users/abhishekkumbhar/Documents/s2-data-processing/run_pipeline.sh
```

---

## 📁 File Structure

```
/Users/abhishekkumbhar/Documents/s2-data-processing/
├── scripts/
│   ├── main_pipeline.py                    # Main pipeline orchestrator
│   ├── step3_daily_combiner.py            # Daily CSV combiner
│   ├── step4_device_bifurcation.py        # Device bifurcation
│   ├── step5_interpolation.py             # OSRM interpolation
│   ├── step6_combine_upload.py            # Regional combiner
│   └── step7_abnormal_events.py           # Abnormal events detection
├── run_pipeline.sh                        # Pipeline wrapper script
├── config/
│   └── pipeline_config.json               # Pipeline configuration
├── logs/                                  # All log files
├── dockerhealthchecks/
│   ├── install_health_check.sh           # Health check installer
│   └── com.seesense.docker.health.daemon.plist
└── AUTOMATION_DOCUMENTATION.md           # This file

/usr/local/bin/seesense/
└── docker_health_check.sh                # Health check script

/Library/LaunchDaemons/
└── com.seesense.docker.health.daemon.plist  # System daemon config

/var/log/seesense/                        # System logs
├── docker_health_check.log               # Health check logs
├── docker_health_daemon.out/.err         # Daemon stdout/stderr
└── pipeline*.log                         # Pipeline logs
```

---

## 🚀 Installation Steps

### **Step 1: Setup Cron Job**
```bash
# Add environment variable and cron job
echo 'SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
30 3 * * * /Users/abhishekkumbhar/Documents/s2-data-processing/run_pipeline.sh' | crontab -

# Verify installation
crontab -l
```

### **Step 2: Install Launch Daemon**
```bash
# Navigate to project directory
cd /Users/abhishekkumbhar/Documents/s2-data-processing

# Run the installer script
./install_daemon.sh

# Test the installation
./test_daemon.sh
```

### **Step 3: Verify System Status**
```bash
# Check cron daemon
launchctl print system/com.vix.cron | grep "state ="

# Check health daemon
sudo launchctl print system/com.seesense.docker.health.daemon

# Check Docker containers
docker ps --format "table {{.Names}}\t{{.Status}}" | grep osrm

# Check logs
tail -f /var/log/seesense/docker_health_check.log
```

---

## 📊 Monitoring & Logs

### **Log Locations**

| Component | Log File | Description |
|-----------|----------|-------------|
| Health Check | `/var/log/seesense/docker_health_check.log` | Container health status |
| Health Daemon | `/var/log/seesense/docker_health_daemon.out` | Daemon stdout |
| Pipeline | `/Users/abhishekkumbhar/Documents/s2-data-processing/logs/` | Pipeline execution logs |
| Cron Runner | `/Users/abhishekkumbhar/Documents/s2-data-processing/logs/daemon_runner.log` | Cron execution logs |

### **Monitoring Commands**
```bash
# Real-time health check monitoring
tail -f /var/log/seesense/docker_health_check.log

# Real-time pipeline monitoring
tail -f /Users/abhishekkumbhar/Documents/s2-data-processing/logs/pipeline_S2_Pipeline.log

# Check daemon status
sudo launchctl print system/com.seesense.docker.health.daemon

# Manual health check test
/usr/local/bin/seesense/docker_health_check.sh test

# Manual pipeline test
/Users/abhishekkumbhar/Documents/s2-data-processing/run_pipeline.sh
```

---

## 📱 Slack Notifications

### **Health Check Notifications**
- **Container failures**: Immediate alerts when containers stop
- **Recovery status**: Notifications when containers restart
- **Daily summary**: Health check completion status

### **Pipeline Notifications**
- **Execution start/end**: Timestamps and duration
- **Success/failure status**: Exit codes and error messages
- **Log snippets**: Last 20 lines of execution logs
- **System information**: Hostname and execution details

### **Notification Format**
```
🩺 Docker Health Check - MacMini
├── Status: ✅ All containers healthy
├── Duration: 00:00:15
└── Containers: finland, ireland, sydney, wales, england

🚀 S2 Pipeline SUCCESS - MacMini
├── Start: 2025-09-25T03:30:00Z
├── End: 2025-09-25T05:45:30Z
├── Duration: 02:15:30
├── Exit Code: 0
└── Last 20 lines: [log snippet]
```

---

## 🛠️ Maintenance & Troubleshooting

### **Common Issues**

#### **Health Daemon Not Running**
```bash
# Check if loaded
sudo launchctl print system/com.seesense.docker.health.daemon

# Reload daemon
sudo launchctl bootout system/com.seesense.docker.health.daemon
sudo launchctl bootstrap system /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist
sudo launchctl enable system/com.seesense.docker.health.daemon
```

#### **Cron Job Not Executing**
```bash
# Check cron daemon
launchctl print system/com.vix.cron

# Verify cron job
crontab -l

# Check execution logs
tail -f /Users/abhishekkumbhar/Documents/s2-data-processing/logs/daemon_runner.log
```

#### **Docker Containers Stopped**
```bash
# Check container status
docker ps -a | grep osrm

# Start all containers
docker start osrm-finland osrm-ireland osrm-sydney osrm-wales osrm-england

# Manual health check
/usr/local/bin/seesense/docker_health_check.sh start-all
```

#### **Slack Notifications Not Working**
```bash
# Test webhook manually
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test notification"}' \
  https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Check environment variable
echo $SLACK_WEBHOOK_URL
```

### **Manual Testing**

#### **Test Health Check**
```bash
# Basic test
/usr/local/bin/seesense/docker_health_check.sh test

# Manual trigger
sudo launchctl kickstart system/com.seesense.docker.health.daemon
```

#### **Test Pipeline**
```bash
# Interactive mode
python /Users/abhishekkumbhar/Documents/s2-data-processing/scripts/main_pipeline.py

# Automated mode
/Users/abhishekkumbhar/Documents/s2-data-processing/run_pipeline.sh

# Specific date
python /Users/abhishekkumbhar/Documents/s2-data-processing/scripts/main_pipeline.py --date 2025-09-24 --automated
```

---

## 🔄 System Updates & Changes

### **Updating Webhook URL**
```bash
# Update Launch Daemon
sudo nano /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist
# Modify SLACK_WEBHOOK_URL value

# Update Cron
crontab -e
# Modify SLACK_WEBHOOK_URL line

# Reload services
sudo launchctl bootout system/com.seesense.docker.health.daemon
sudo launchctl bootstrap system /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist
```

### **Changing Schedule**
```bash
# Modify health check time (Launch Daemon)
sudo nano /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist
# Change Hour/Minute values

# Modify pipeline time (Cron)
crontab -e
# Change time values (minute hour * * *)

# Reload services
sudo launchctl bootout system/com.seesense.docker.health.daemon
sudo launchctl bootstrap system /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist
```

### **Adding New Containers**
```bash
# Edit health check script
sudo nano /usr/local/bin/seesense/docker_health_check.sh
# Modify CONTAINER_NAMES and CONTAINER_PORTS arrays

# Test changes
/usr/local/bin/seesense/docker_health_check.sh test
```

---

## 📈 System Requirements

### **Hardware**
- Mac Mini (or any macOS system)
- Minimum 8GB RAM (16GB recommended for data processing)
- 100GB+ free disk space for data and logs
- Reliable internet connection

### **Software Dependencies**
- **macOS**: Monterey 12.0+ (for Launch Daemon compatibility)
- **Docker Desktop**: 4.0+
- **Conda**: Miniconda or Anaconda
- **Python**: 3.8+ (in conda environment)
- **Bash**: 5.0+ (default macOS bash is fine)

### **Network Requirements**
- Access to S3 buckets for data retrieval
- OSRM routing service access
- Slack webhook connectivity
- Docker registry access (for container updates)

---

## 🔒 Security Considerations

### **File Permissions**
- Launch Daemon plist: `644 root:wheel`
- Health check script: `755 root:wheel`
- Pipeline scripts: `755 user:staff`
- Log directories: `755 user:staff`

### **Network Security**
- Webhook URLs contain sensitive tokens
- S3 credentials stored in conda environment
- Docker containers run with user privileges (not root)

### **Access Control**
- System-level daemons run with minimal privileges
- User-level cron jobs run as specific user
- Log files readable by user only

---

#### **Complete System Reset**
```bash
# Stop all automation
sudo launchctl bootout system/com.seesense.docker.health.daemon
crontab -r

# Remove all automation files
sudo rm /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist
rm -rf /Users/abhishekkumbhar/Documents/s2-data-processing/install_daemon.sh
rm -rf /Users/abhishekkumbhar/Documents/s2-data-processing/test_daemon.sh

# Restart Docker containers manually
docker restart $(docker ps -aq --filter "name=osrm-")
```

#### **Backup Critical Files**
```bash
# Configuration backup
cp /Library/LaunchDaemons/com.seesense.docker.health.daemon.plist ~/backup/
crontab -l > ~/backup/crontab_backup.txt
cp /usr/local/bin/seesense/docker_health_check.sh ~/backup/

# Pipeline backup
tar -czf ~/backup/s2-pipeline-$(date +%Y%m%d).tar.gz \
  /Users/abhishekkumbhar/Documents/s2-data-processing/
```

---


## ✅ Verification Checklist

Use this checklist to verify the automation setup:

### **Pre-flight Check**
- [ ] Docker Desktop installed and running
- [ ] All 5 OSRM containers present and healthy
- [ ] Conda environment `s2-data-processing` exists
- [ ] Slack webhook URL obtained and tested
- [ ] System has sufficient disk space (>100GB free)

### **Installation Verification**
- [ ] Launch Daemon plist installed in `/Library/LaunchDaemons/`
- [ ] Health check script installed in `/usr/local/bin/seesense/`
- [ ] Cron job configured with environment variable
- [ ] Log directories created with proper permissions
- [ ] Test scripts (`install_daemon.sh`, `test_daemon.sh`) working

### **Runtime Verification**
- [ ] Launch Daemon loaded and scheduled
- [ ] Cron daemon running
- [ ] Health check script executable and functional
- [ ] Pipeline script executable and functional
- [ ] Slack notifications working for both systems

### **Monitoring Setup**
- [ ] Log files rotating properly
- [ ] Health check logs showing container status
- [ ] Pipeline logs showing execution details
- [ ] Slack notifications received for test runs

---

*This documentation serves as a complete reference for the S2 data processing pipeline automation. Keep this file updated when making system changes.*