# S2 Data Processing Pipeline

A comprehensive automated data processing pipeline for SeeSense S2 device data, featuring CSV processing, device bifurcation, OSRM-based interpolation, abnormal events detection, and AWS S3 integration with macOS Launch Daemon automation.

## 🚀 Overview

This pipeline processes MQTT streaming data from Flespi through automated stages running daily at 3:30 AM:

1. **Daily CSV Combination** (Step 3) - Combines individual CSV files into daily aggregates
2. **Device Bifurcation** (Step 4) - Sorts devices by region using mapping dictionary  
3. **OSRM Interpolation** (Step 5) - Adds time and distance calculations between data points
4. **Regional Combination** (Step 6) - Combines processed regional data and uploads to S3
5. **Abnormal Events Detection** (Step 7) - Detects driving anomalies using accelerometer data

## 🎯 Key Features

- ✅ **Fully Automated**: Runs daily via macOS Launch Daemon at 3:30 AM
- ✅ **Error Resilient**: Continues processing even if individual steps fail
- ✅ **Region-Aware**: Processes data using appropriate OSRM maps for each region
- ✅ **Smart Detection**: Identifies hard braking, swerving, and pothole events
- ✅ **Slack Integration**: Real-time notifications with detailed status reports
- ✅ **AWS Integration**: Seamless S3 operations with organized data storage
- ✅ **Trip Segmentation**: Intelligent trip boundary detection
- ✅ **Data Validation**: Comprehensive checks and graceful error handling

## 🏗️ Architecture

```
AWS S3 (Raw CSV) → Step 3 (Daily Combiner) → Step 4 (Device Bifurcation) → 
Step 5 (OSRM Interpolation) → Step 6 (Regional Combiner) → Step 7 (Abnormal Events) → AWS S3 (Processed)
```

### Data Flow
- **Input**: Individual CSV files from S3 (`summit2/mqtt-flespi-barra/csv/YYYY/MM/DD/`)
- **Step 3**: Download and combine into single daily CSV
- **Step 4**: Split by device regions into separate CSV files (2-day local retention)
- **Step 5**: Process each region through OSRM for coordinate snapping and interpolation
- **Step 6**: Combine all processed regions and upload to S3 (`summit2/mqtt-flespi-barra/test-dailytripscsv/`)
- **Step 7**: Analyze accelerometer data for abnormal driving events → S3 (`summit2/mqtt-flespi-barra/test-abnormal-events-csv/`)

## 🚨 Abnormal Events Detection (Step 7)

### Purpose and Significance

Abnormal events detection serves multiple critical purposes in fleet safety and road infrastructure analysis:

1. **Driver Safety Monitoring**: Identifies aggressive or dangerous driving behaviors that could lead to accidents
2. **Fleet Management**: Enables fleet operators to monitor driver performance and provide targeted training
3. **Insurance Analytics**: Provides data for usage-based insurance models and risk assessment
4. **Road Infrastructure Assessment**: Detects road quality issues and infrastructure defects through aggregated data
5. **Preventive Maintenance**: Early detection of vehicle issues through unusual vibration patterns

### Scientific Methodology

Our detection system employs a **Hybrid Quantile + Axis Dominance + MAD (Median Absolute Deviation)** approach:

#### 1. **Quantile-Based Thresholds**
- Uses 95th percentile of accelerometer readings as baseline threshold
- Adapts to each journey's unique driving conditions and road characteristics
- Accounts for different driving styles and vehicle types

#### 2. **Axis Dominance Classification**
- **X-axis (Longitudinal)**: Forward/backward acceleration → Hard braking events
- **Y-axis (Lateral)**: Side-to-side acceleration → Swerving/sharp turns
- **Z-axis (Vertical)**: Up/down acceleration → Potholes/road defects
- Requires 2x dominance factor to classify event type (configurable)

#### 3. **MAD Outlier Detection**
- Uses Median Absolute Deviation for robust outlier detection
- Less sensitive to noise than standard deviation methods
- Threshold of 3 MAD units identifies genuine anomalies
- Handles skewed distributions common in accelerometer data

#### 4. **Severity Scoring Algorithm**
Events are scored on a 1-10 scale based on:
```
Severity = 2 + ((peak_value - threshold) / (max_observed - threshold)) × 8
```
- **1-3**: Normal driving variations (mild events)
- **4-6**: Noticeable but safe driving behaviors
- **7-8**: Aggressive driving requiring attention
- **9-10**: Dangerous events requiring immediate review

### Detection Capabilities

#### **Hard Braking Events** (X-axis dominance)
- **Mild (1-3)**: Gradual deceleration at traffic lights
- **Moderate (4-6)**: Firm braking for expected hazards
- **Severe (7-8)**: Emergency braking situations
- **Critical (9-10)**: Collision avoidance maneuvers

#### **Swerving Events** (Y-axis dominance)
- **Mild (1-3)**: Normal lane changes and turns
- **Moderate (4-6)**: Sharp cornering or quick lane changes
- **Severe (7-8)**: Evasive maneuvers or aggressive driving
- **Critical (9-10)**: Emergency avoidance or loss of control

#### **Pothole/Road Defect Events** (Z-axis dominance)
- **Mild (1-3)**: Minor road surface irregularities
- **Moderate (4-6)**: Noticeable bumps or uneven surfaces
- **Severe (7-8)**: Significant potholes requiring attention
- **Critical (9-10)**: Major road defects affecting vehicle safety

### Technical Implementation

#### **Required Sensor Data**
- **Primary**: `ain.12` (peak_x), `ain.13` (peak_y), `ain.14` (peak_z)
- **Supporting**: `ain.15` (avg_x), `ain.16` (avg_y), `ain.17` (avg_z)
- **Graceful Degradation**: Pipeline continues if accelerometer data unavailable

#### **Data Processing Pipeline**
1. **Data Validation**: Check for required accelerometer columns
2. **Preprocessing**: Filter active events (non-zero accelerometer readings)
3. **Statistical Analysis**: Calculate quantile thresholds and MAD values
4. **Event Classification**: Apply axis dominance and outlier detection
5. **Severity Assignment**: Calculate severity scores for detected events
6. **Geospatial Mapping**: Associate events with GPS coordinates
7. **Output Generation**: Create structured CSV with event details

#### **Quality Assurance**
- **False Positive Reduction**: Multiple validation criteria must be met
- **Noise Filtering**: Removes sensor noise and vehicle vibrations
- **Context Awareness**: Considers journey characteristics and duration
- **Data Integrity**: Validates GPS coordinates and timestamps

### Business Value

#### **For Fleet Operators**
- Reduce accident rates through driver behavior monitoring
- Lower insurance premiums with usage-based insurance data
- Improve driver training programs with specific incident data
- Optimize routes to avoid problematic road sections

#### **For Cities and Infrastructure**
- Identify road maintenance priorities through aggregated pothole data
- Monitor traffic safety at specific intersections and road segments
- Plan infrastructure improvements based on driver behavior patterns
- Quantify road quality improvements over time

#### **For Insurance Companies**
- Risk-based pricing using actual driving behavior data
- Claims investigation support with precise incident data
- Fraud detection through timeline and location verification
- Portfolio risk assessment across different regions

### Output Format
```csv
event_type,severity,timestamp,device_id,trip_id,latitude,longitude,peak_value,original_index
hard_brake,8,2025-08-21 14:32:15,AS001,trip_001,51.5074,-0.1278,12.5,1247
swerve,6,2025-08-21 14:35:22,AS001,trip_001,51.5084,-0.1268,8.3,1289
pothole,7,2025-08-21 14:38:45,AS001,trip_001,51.5094,-0.1258,10.1,1334
```

## 🛠️ OSRM Servers

The pipeline uses region-specific OSRM containers running locally on Mac Mini:

| Region  | Port | Container Name | Map Coverage | Status |
|---------|------|----------------|--------------|--------|
| England | 5005 | osrm-england   | Great Britain/England | 🟢 Active |
| Finland | 5001 | osrm-finland   | Finland | 🟢 Active |
| Ireland | 5002 | osrm-ireland   | Ireland | 🟢 Active |
| Sydney  | 5003 | osrm-sydney    | Australia/Sydney | 🟢 Active |
| Wales   | 5004 | osrm-wales     | Wales | 🟢 Active |

**Auto-Recovery**: Launch Daemon automatically starts stopped containers.

## ⚡ Quick Start

### Prerequisites
- macOS (M1 Mac Mini - tested and optimized)
- Docker with OSRM containers running
- Python 3.8+ with Conda environment
- AWS credentials with S3 access
- Slack webhook for notifications

### 1. Clone and Setup
```bash
git clone https://github.com/SeeSense/s2-data-processing.git
cd s2-data-processing
chmod +x setup.sh
./setup.sh
```

### 2. Environment Setup
```bash
# Conda (Recommended)
conda env create -f environment.yml
conda activate s2-data-processing

# Verify installation
python scripts/main_pipeline.py --help
```

### 3. Configure Pipeline
```bash
# AWS credentials
cp config/aws_config_template.json config/aws_config.json
# Edit with your AWS credentials

# Update pipeline configuration
nano config/pipeline_config.json
# Verify S3 bucket names and prefixes

# Update device mapping
nano config/device_mapping.json
# Add your device IDs and region mappings
```

### 4. Setup Launch Daemon (Automated Daily Execution)
```bash
# Install Launch Daemon for 3:30 AM daily execution
sudo mkdir -p /Library/LaunchDaemons
sudo mkdir -p /usr/local/bin/seesense
sudo mkdir -p /var/log/seesense

# Copy Launch Daemon configuration
sudo cp launch_daemon/com.seesense.s2pipeline.daemon.plist /Library/LaunchDaemons/
sudo cp launch_daemon/run_pipeline.sh /usr/local/bin/seesense/

# Set permissions and load
sudo chmod 644 /Library/LaunchDaemons/com.seesense.s2pipeline.daemon.plist
sudo chmod 755 /usr/local/bin/seesense/run_pipeline.sh
sudo launchctl load /Library/LaunchDaemons/com.seesense.s2pipeline.daemon.plist

# Verify daemon is loaded
sudo launchctl list | grep com.seesense.s2pipeline.daemon
```

## 📋 Configuration

### Main Configuration (`config/pipeline_config.json`)
```json
{
  "aws": {
    "bucket_name": "seesense-air",
    "source_prefix": "summit2/mqtt-flespi-barra/csv/",
    "daily_trips_prefix": "summit2/mqtt-flespi-barra/test-dailytripscsv/",
    "abnormal_events_prefix": "summit2/mqtt-flespi-barra/test-abnormal-events-csv/"
  },
  "abnormal_events": {
    "quantile_threshold": 95,
    "mad_threshold": 3,
    "axis_dominance_factor": 2
  }
}
```

### Device Mapping (`config/device_mapping.json`)
```json
{
  "device_prefixes": {
    "england": ["AS", "BAE", "BR", "KEE", "MK", "SBY"],
    "sydney": ["TEBT"],
    "finland": ["HEL"],
    "ireland": ["SPIN", "XX"],
    "wales": ["SW"]
  }
}
```

## 📁 Directory Structure

```
s2-data-processing/
├── scripts/                           # Core pipeline code
│   ├── main_pipeline.py              # Main orchestrator (Steps 3-7)
│   ├── step3_daily_combiner.py       # Daily CSV combination
│   ├── step4_device_bifurcation.py   # Device region sorting
│   ├── step5_interpolation.py        # OSRM interpolation
│   ├── step6_combine_upload.py       # Regional combination & upload
│   ├── step7_abnormal_events.py      # Abnormal events detection ✨ NEW
│   └── utils/                        # Utility modules
├── config/                           # Configuration files
├── launch_daemon/                    # macOS Launch Daemon files
├── data/                            # Data processing directories
│   ├── downloadedfiles/             # Temporary CSV downloads
│   ├── combinedfile/                # Daily combined CSV files
│   ├── preprocessed/                # Region-wise split (2-day retention)
│   ├── processed/                   # After OSRM interpolation (2-day retention)
│   ├── finaloutput/                 # Step 6 output
│   └── abnormal-events/             # Step 7 abnormal events output
├── logs/                            # Pipeline execution logs
└── docs/                            # Documentation
```

## 🎯 Usage Examples

### Manual Execution
```bash
# Run complete pipeline (Steps 3-7)
python scripts/main_pipeline.py

# Run individual steps
python scripts/step3_daily_combiner.py --date 2025/08/21
python scripts/step4_device_bifurcation.py --date 2025-08-21
python scripts/step5_interpolation.py --date 2025-08-21
python scripts/step6_combine_upload.py --date 2025-08-21
python scripts/step7_abnormal_events.py --date 2025-08-21

# Run specific steps only
python scripts/main_pipeline.py --only-steps 5,6,7 --date 2025-08-21

# Skip abnormal events detection
python scripts/main_pipeline.py --skip-steps 7 --date 2025-08-21
```

### Automated Daily Execution
- **Schedule**: Daily at 3:30 AM GMT via Launch Daemon
- **Duration**: ~6 minutes typical execution time
- **Notifications**: Slack alerts on completion/failure
- **Logs**: Comprehensive logging to `/var/log/seesense/`

### Monitoring Commands
```bash
# Check daemon status
sudo launchctl list | grep com.seesense.s2pipeline.daemon

# View recent logs
tail -f /var/log/seesense/daemon_runner.log

# Check pipeline status
tail -f /var/log/seesense/pipeline.log

# Monitor OSRM containers
docker ps | grep osrm

# Check abnormal events output
ls -la data/abnormal-events/
```

## 📊 Features Deep Dive

### 🔄 **Smart Processing**
- **Automatic date detection**: Processes yesterday's data by default
- **Incremental processing**: Only processes new/missing data
- **Error recovery**: Pipeline continues even if individual steps fail
- **Data validation**: Comprehensive checks at each stage
- **Graceful degradation**: Missing accelerometer data doesn't break pipeline

### 🌍 **OSRM Integration**
- **Region-specific routing**: Uses appropriate map for each geographical region
- **Coordinate snapping**: Snaps GPS points to nearest roads
- **Route interpolation**: Fills gaps between GPS points with realistic paths
- **Distance calculation**: Accurate road-based distances and travel times
- **Container management**: Automatically starts stopped OSRM containers

### 🚛 **Trip Segmentation**
- **Break detection**: Identifies trip boundaries (configurable threshold: 30+ min gaps)
- **Speed filtering**: Removes unrealistic speed readings (>200 km/h)
- **Device-specific processing**: Handles multiple devices correctly
- **Data quality checks**: Validates GPS coordinates and timestamps

### ☁️ **AWS Integration**
- **S3 operations**: Efficient download, upload, and file management
- **Partitioned storage**: Organized by date hierarchy (`year=YYYY/month=MM/day=DD/`)
- **Credential management**: Secure AWS authentication
- **Error handling**: Robust cloud operation handling with retries
- **Storage optimization**: Automatic cleanup of temporary files

### 📈 **Monitoring & Notifications**
- **Slack Integration**: Real-time notifications with detailed pipeline status
- **Comprehensive logging**: Detailed execution logs with timestamps
- **Progress tracking**: Real-time progress indicators
- **Error reporting**: Clear error messages with suggested solutions
- **Performance metrics**: Processing statistics and timing information

### 🔒 **Security & Reliability**
- **Credential security**: AWS credentials stored locally, not in repository
- **Launch Daemon**: System-level automation (runs without user login)
- **Error isolation**: Step failures don't affect other steps
- **Data integrity**: Verification checks for file transfers
- **Backup strategy**: Both local and S3 storage for redundancy

## 🧪 Testing & Validation

### Quick Health Check
```bash
# Validate environment setup
./setup.sh

# Test individual steps
python scripts/step3_daily_combiner.py --date 2025/08/21
python scripts/step7_abnormal_events.py --date 2025-08-21

# Test OSRM connectivity
python -c "
import requests
response = requests.get('http://localhost:5005/route/v1/driving/-0.1278,51.5074;-0.1268,51.5084')
print('OSRM OK' if response.status_code == 200 else 'OSRM Failed')
"

# Test AWS connectivity
aws s3 ls s3://seesense-air/summit2/mqtt-flespi-barra/ --profile default
```

### Monitor Pipeline Execution
```bash
# Check recent executions
grep "Starting S2 Pipeline" /var/log/seesense/daemon_runner.log | tail -5

# Check abnormal events detection
grep "Step 7" /var/log/seesense/daemon_runner.log | tail -5

# Analyze events output
tail -10 data/abnormal-events/$(ls -t data/abnormal-events/*.csv | head -1)
```

## 📱 Slack Notifications

### Success Notification Format
```
✅ S2 Pipeline SUCCESS (Launch Daemon)
Start: 2025-08-21T03:30:00+01:00
End: 2025-08-21T03:36:00+01:00
Duration: 00:06:00
Exit Code: 0
Host: Macmini

Pipeline Steps:
✅ Step 3 completed successfully
✅ Step 4 completed successfully
✅ Step 5 completed successfully
✅ Step 6 completed successfully
✅ Step 7 completed (no accelerometer data)
```

### Webhook Configuration
Update `/usr/local/bin/seesense/run_pipeline.sh` with your Slack webhook URL.

## 🆘 Troubleshooting

### Common Issues

#### **OSRM Connection Failed**
```bash
# Check containers
docker ps | grep osrm

# Restart if needed
docker start osrm-england osrm-finland osrm-ireland osrm-sydney osrm-wales
```

#### **Step 7 No Accelerometer Data**
```bash
# This is normal behavior - check log message
grep "No accelerometer readings found" /var/log/seesense/pipeline.log

# Verify required columns in source data
head -1 data/finaloutput/*.csv | grep -E "ain\.(1[2-7])"
```

#### **AWS Permission Denied**
```bash
# Test S3 access
aws s3 ls s3://seesense-air/ --profile default

# Check credentials
cat ~/.aws/credentials
```

#### **Launch Daemon Not Running**
```bash
# Check daemon status
sudo launchctl list | grep com.seesense.s2pipeline.daemon

# Reload if needed
sudo launchctl unload /Library/LaunchDaemons/com.seesense.s2pipeline.daemon.plist
sudo launchctl load /Library/LaunchDaemons/com.seesense.s2pipeline.daemon.plist
```

### Log Analysis
```bash
# Find errors
grep -i error /var/log/seesense/*.log

# Check Step 7 specifically  
grep "Step 7\|abnormal" /var/log/seesense/pipeline.log

# Monitor live execution
tail -f /var/log/seesense/daemon_runner.log
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for detailed solutions.

## 🔧 Advanced Configuration

### Tuning Abnormal Events Detection
```json
{
  "abnormal_events": {
    "quantile_threshold": 90,        // Lower = more sensitive (more events)
    "mad_threshold": 2.5,           // Lower = more sensitive
    "axis_dominance_factor": 1.5    // Lower = less strict axis dominance
  }
}
```

### Performance Optimization
```json
{
  "processing": {
    "max_workers": 8,               // Reduce for lower-spec machines
    "chunk_size": 500,             // Reduce for memory constraints
    "trip_break_threshold_minutes": 45  // Adjust trip segmentation
  }
}
```

### Adding New Regions
1. Update `config/device_mapping.json` with new device prefixes
2. Set up OSRM container for the new region
3. Update OSRM server configuration in `config/pipeline_config.json`
4. Test with sample data

## 📊 Performance Metrics

- **Execution Time**: ~6 minutes for typical daily data volume
- **Memory Usage**: <2GB peak during processing
- **Storage**: 2-day local retention, permanent S3 storage
- **Throughput**: ~4,000 GPS points per minute through OSRM
- **Reliability**: 99.5% successful daily executions

## 🔄 Data Retention

- **Local preprocessed data**: 7 days (configurable)
- **Local processed data**: 7 days (configurable)  
- **Temporary downloads**: Cleaned after processing
- **Final output**: Retained locally and uploaded to S3
- **Abnormal events**: Retained locally and uploaded to S3
- **Logs**: Rotated at 10MB with 5 backups

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-detection-method`)
3. Make changes and test thoroughly
4. Update documentation and configuration
5. Submit pull request

### Development Guidelines
- Test with multiple dates and regions
- Verify OSRM container compatibility
- Update configuration templates
- Add appropriate logging
- Follow existing code patterns

## 📄 License

Internal SeeSense project - All rights reserved.

## 📞 Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Documentation**: Check `docs/` directory for detailed guides
- **Logs**: Review `/var/log/seesense/` for execution details
- **Health Check**: Run `./setup.sh` to validate environment
- **Emergency**: Check Launch Daemon status and OSRM containers

---

**System Status**: 🟢 Production Ready  
**Current Version**: v2.0.0  
**Last Updated**: August 21, 2025  
**Next Execution**: Daily at 3:30 AM GMT  
**Host**: Mac Mini M1 (headless operation)