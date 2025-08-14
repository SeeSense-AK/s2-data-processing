# S2 Data Processing Pipeline

A comprehensive data processing pipeline for SeeSense S2 device data, featuring automated CSV processing, device bifurcation, OSRM-based interpolation, and AWS S3 integration.

## 🚀 Overview

This pipeline processes MQTT streaming data from Flespi through several automated stages:

1. **Daily CSV Combination** - Combines individual CSV files into daily aggregates
2. **Device Bifurcation** - Sorts devices by region using mapping dictionary  
3. **OSRM Interpolation** - Adds time and distance calculations between data points
4. **Regional Combination** - Combines processed regional data and uploads to S3

## 🏗️ Architecture

```
AWS S3 (CSV Files) → Daily Combiner → Device Bifurcation → OSRM Interpolation → Regional Combiner → AWS S3 (Processed)
```

### Data Flow
- **Input**: Individual CSV files from S3 (`summit2/mqtt-flespi-barra/csv/YYYY/MM/DD/`)
- **Step 3**: Download and combine into single daily CSV
- **Step 4**: Split by device regions into separate CSV files
- **Step 5**: Process each region through OSRM for interpolation
- **Step 6**: Combine all processed regions and upload to S3 (`summit2/mqtt-flespi-barra/dailytripscsv/`)

## 🛠️ OSRM Servers

The pipeline uses region-specific OSRM containers running locally:

| Region  | Port | Container Name | Map Coverage |
|---------|------|----------------|--------------|
| England | 5005 | osrm-england   | Great Britain/England |
| Finland | 5001 | osrm-finland   | Finland |
| Ireland | 5002 | osrm-ireland   | Ireland |
| Sydney  | 5003 | osrm-sydney    | Australia/Sydney |
| Wales   | 5004 | osrm-wales     | Wales |

## ⚡ Quick Start

### Prerequisites
- macOS (tested on M1 Mac Mini)
- Docker with OSRM containers running
- Python 3.8+ or Conda
- AWS credentials with S3 access

### 1. Clone and Setup
```bash
git clone https://github.com/YourOrg/s2-data-processing.git
cd s2-data-processing
chmod +x setup.sh
./setup.sh
```

### 2. Environment Setup
```bash
# Option A: Conda (Recommended)
conda env create -f environment.yml
conda activate s2-data-processing

# Option B: pip
pip install -r requirements.txt
```

### 3. Configure Credentials
```bash
cp config/aws_config_template.json config/aws_config.json
# Edit aws_config.json with your AWS credentials
```

### 4. Run Pipeline
```bash
# Interactive mode
python scripts/main_pipeline.py

# Automated mode (for cron)
python scripts/main_pipeline.py --automated

# Process specific date
python scripts/main_pipeline.py --date 2025-08-13
```

## 📋 Configuration

### Main Configuration (`config/pipeline_config.json`)
- AWS S3 bucket settings
- OSRM server endpoints  
- Processing parameters
- Directory paths
- Scheduling settings

### Device Mapping (`config/device_mapping.json`)
- Device ID to region mapping
- Region-specific settings
- Timezone configurations

Example device mapping:
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
├── scripts/                    # Core pipeline code
│   ├── main_pipeline.py       # Main orchestrator
│   ├── step3_daily_combiner.py
│   ├── step4_device_bifurcation.py
│   ├── step5_interpolation.py
│   ├── step6_combine_upload.py
│   └── utils/                 # Utility modules
├── config/                    # Configuration files
├── tests/                     # Test scripts
├── data/                      # Data processing directories
│   ├── downloadedfiles/       # Temporary CSV downloads
│   ├── combinedfile/          # Daily combined CSV files
│   ├── preprocessed/          # Region-wise split data (2-day retention)
│   └── processed/             # After OSRM interpolation
├── logs/                      # Pipeline logs
└── docs/                      # Documentation
```

## 🔧 Usage Examples

### Manual Execution
```bash
# Run complete pipeline
python scripts/main_pipeline.py

# Run individual steps
python scripts/step3_daily_combiner.py --date 2025/08/13
python scripts/step4_device_bifurcation.py --date 2025-08-13
python scripts/step5_interpolation.py --date 2025-08-13
python scripts/step6_combine_upload.py --date 2025-08-13

# Run specific steps only
python scripts/main_pipeline.py --only-steps 5,6 --date 2025-08-13

# Skip certain steps
python scripts/main_pipeline.py --skip-steps 3,4
```

### Automated Scheduling
```bash
# Add to crontab for daily execution at 3:30 AM
30 3 * * * cd /path/to/s2-data-processing && python scripts/main_pipeline.py --automated
```

## 🧪 Testing

### Validate Setup
```bash
# Test Step 3 (Daily Combiner)
python tests/test_step3.py

# Test Step 5 (OSRM Interpolation)
python tests/test_step5.py
```

### Environment Health Check
```bash
# Run setup script to validate environment
./setup.sh
```

## 📊 Features

### 🔄 **Smart Processing**
- **Automatic date detection**: Finds unprocessed dates
- **Incremental processing**: Only processes new data
- **Error recovery**: Continues processing even if one step fails
- **Data validation**: Comprehensive checks at each stage

### 🌍 **OSRM Integration**
- **Region-specific routing**: Uses appropriate map for each region
- **Coordinate snapping**: Snaps GPS points to roads
- **Route interpolation**: Fills gaps between GPS points
- **Distance calculation**: Accurate road-based distances

### 🚛 **Trip Segmentation**
- **Break detection**: Identifies trip boundaries (>30 min gaps)
- **Speed filtering**: Removes unrealistic speed readings
- **Device-specific processing**: Handles multiple devices correctly

### ☁️ **AWS Integration**
- **S3 operations**: Download, upload, and manage files
- **Partitioned storage**: Organized by date hierarchy
- **Credential management**: Secure AWS authentication
- **Error handling**: Robust cloud operation handling

### 📈 **Monitoring & Logging**
- **Comprehensive logging**: Detailed execution logs
- **Progress tracking**: Real-time progress indicators
- **Error reporting**: Clear error messages and solutions
- **Performance metrics**: Processing statistics and timing

## 🔧 Advanced Usage

### Development Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/main_pipeline.py

# Process single region
python scripts/step5_interpolation.py --region england --date 2025-08-13

# Force reprocessing (overwrite existing)
python scripts/step6_combine_upload.py --force --date 2025-08-13
```

### Production Monitoring
```bash
# Check pipeline status
tail -f logs/pipeline.log

# Monitor OSRM servers
docker ps | grep osrm

# Check data freshness
find data/processed -name "*.csv" -mtime -1
```

## 🆘 Troubleshooting

### Common Issues

1. **OSRM Connection Failed**: Check Docker containers are running
   ```bash
   docker ps | grep osrm
   docker start osrm-england  # if needed
   ```

2. **AWS Permission Denied**: Verify S3 bucket permissions
   ```bash
   aws s3 ls s3://your-bucket/  # test access
   ```

3. **No Data Found**: Check S3 path and date format
   ```bash
   python scripts/step3_daily_combiner.py --date 2025/08/12  # try different date
   ```

4. **Memory Issues**: Reduce processing batch size
   ```json
   // In config/pipeline_config.json
   "processing": {
     "chunk_size": 500,
     "max_workers": 5
   }
   ```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for detailed solutions.

## 📚 Documentation

- [**Setup Guide**](docs/SETUP.md) - Detailed installation instructions
- [**Configuration Guide**](docs/CONFIGURATION.md) - Complete configuration reference
- [**Troubleshooting**](docs/TROUBLESHOOTING.md) - Common issues and solutions

## 🛡️ Security

- ✅ AWS credentials stored locally (not in Git)
- ✅ Configuration templates provided
- ✅ Sensitive data excluded from repository
- ✅ Secure S3 operations with IAM roles

## 🔄 Data Retention

- **Local preprocessed data**: 7 days (configurable)
- **Temporary downloads**: Cleaned after processing
- **Logs**: Rotated at 10MB with 5 backups
- **S3 processed data**: Permanent storage

## 🚀 Performance

- **Concurrent processing**: Up to 10 parallel downloads
- **Memory efficient**: Streaming CSV processing
- **Network optimized**: Compressed transfers
- **Disk efficient**: Automatic cleanup

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-region`)
3. Make changes and test thoroughly
4. Update documentation if needed
5. Submit pull request

### Adding New Regions
1. Update `config/device_mapping.json`
2. Set up OSRM container for new region
3. Update port configuration
4. Test with sample data

## 📄 License

Internal SeeSense project - All rights reserved.

## 📞 Support

- **Issues**: Use GitHub Issues for bug reports
- **Documentation**: Check `docs/` directory
- **Logs**: Review `logs/pipeline.log` for errors
- **Health Check**: Run `./setup.sh` to validate environment
