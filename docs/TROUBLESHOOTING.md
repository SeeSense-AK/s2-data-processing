# S2 Data Pipeline - Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 🐳 Docker/OSRM Issues

#### **OSRM Container Not Running**
```bash
# Error: Connection refused on port 5005
❌ england: Connection refused on port 5005 (container: osrm-england)

# Solution: Start the container
docker start osrm-england

# Check if running
docker ps | grep osrm-england

# If container doesn't exist, recreate it
docker run -d --name osrm-england -p 5005:5000 ghcr.io/project-osrm/osrm-backend osrm-routed /data/england-latest.osrm
```

#### **Wrong OSRM Port Configuration**
```bash
# Error: OSRM connection test failed
❌ finland: Error testing server - Connection refused

# Check actual container ports
docker ps | grep osrm

# Update config/pipeline_config.json with correct ports
```

#### **Docker Not Running**
```bash
# Error: Cannot connect to the Docker daemon
❌ Docker connection failed

# Solution: Start Docker
# macOS: Open Docker Desktop
# Linux: sudo systemctl start docker
```

### 🔑 AWS/S3 Issues

#### **AWS Credentials Not Found**
```bash
# Error: AWS credentials not found
❌ AWS connection failed: Unable to locate credentials

# Solution 1: Check config file exists
ls -la config/aws_config.json

# Solution 2: Create from template
cp config/aws_config_template.json config/aws_config.json
# Edit with real credentials

# Solution 3: Use environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

#### **S3 Permission Denied**
```bash
# Error: Access denied to S3 bucket
❌ AWS connection error: AccessDenied

# Solution: Check IAM permissions
# Required policies: s3:GetObject, s3:PutObject, s3:ListBucket
# On bucket: arn:aws:s3:::your-bucket/*
```

#### **S3 Bucket Not Found**
```bash
# Error: Bucket does not exist
❌ Bucket access confirmed: {'exists': False}

# Solution: Verify bucket name in config
# Check config/pipeline_config.json -> aws.bucket_name
```

### 📁 File/Directory Issues

#### **No CSV Files Found**
```bash
# Error: No files found for date
❌ No files found for date 2025-08-13
📂 Checked path: s3://bucket/summit2/mqtt-flespi-barra/csv/2025/08/13/

# Solution 1: Check S3 path structure
aws s3 ls s3://your-bucket/summit2/mqtt-flespi-barra/csv/ --recursive

# Solution 2: Try different date
python scripts/step3_daily_combiner.py --date 2025/08/12

# Solution 3: Check source_prefix in config
```

#### **Permission Denied on Local Directories**
```bash
# Error: Permission denied
❌ PermissionError: [Errno 13] Permission denied: '/data/downloadedfiles'

# Solution: Fix permissions
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

#### **No Preprocessed Data**
```bash
# Error: No preprocessed data found
❌ No preprocessed data available for testing

# Solution: Run Step 4 first
python scripts/step4_device_bifurcation.py --date 2025-08-13
```

### 🧭 Device Mapping Issues

#### **Device Not Mapped to Region**
```bash
# Error: No region found for device
⚠️ No region found for device: ABC123

# Solution: Add device prefix to config/device_mapping.json
{
  "device_prefixes": {
    "england": ["AS", "BAE", "BR", "ABC"],  // Add "ABC"
    // ...
  }
}
```

#### **Wrong Region Assignment**
```bash
# Error: Device in wrong region
❌ Device LON001 assigned to ireland instead of england

# Solution: Update device mapping
# Move "LON" prefix from ireland to england in device_mapping.json
```

### 🔄 Processing Issues

#### **Memory Issues with Large Files**
```bash
# Error: MemoryError during CSV processing
❌ MemoryError: Unable to allocate array

# Solution 1: Reduce chunk_size in config
"processing": {
  "chunk_size": 500,  // Reduce from 1000
  "max_workers": 5    // Reduce concurrent operations
}

# Solution 2: Process smaller date ranges
python scripts/main_pipeline.py --date 2025-08-13 --only-steps 4,5
```

#### **Trip Break Threshold Too Sensitive**
```bash
# Error: Too many trip breaks detected
ℹ️ Trip breaks detected: 5000

# Solution: Increase threshold in config
"processing": {
  "trip_break_threshold_minutes": 60  // Increase from 30
}
```

#### **Coordinate Snapping Failures**
```bash
# Error: Most coordinates fail to snap
❌ Failed to snap coordinate (0.0, 0.0): Invalid coordinate

# Solution: Check data quality
# Filter out zero coordinates before processing
# Verify OSRM server has correct map data
```

### 🐍 Python/Environment Issues

#### **Module Not Found**
```bash
# Error: No module named 'pandas'
ModuleNotFoundError: No module named 'pandas'

# Solution 1: Install requirements
pip install -r requirements.txt

# Solution 2: Activate conda environment
conda activate s2-data-processing

# Solution 3: Install missing package
conda install pandas
```

#### **Python Path Issues**
```bash
# Error: No module named 'scripts.utils'
ModuleNotFoundError: No module named 'scripts.utils'

# Solution: Run from project root
cd /path/to/s2-data-processing
python scripts/main_pipeline.py
```

#### **Permission Denied on Script**
```bash
# Error: Permission denied when running script
❌ bash: ./setup.sh: Permission denied

# Solution: Make executable
chmod +x setup.sh
```

### 📊 Data Quality Issues

#### **Empty CSV Files**
```bash
# Error: CSV file is empty
❌ Empty CSV file: region_20250813.csv

# Solution: Check source data
# Verify Step 3 completed successfully
# Check if devices were active on that date
```

#### **Invalid Date Format**
```bash
# Error: Invalid date format
❌ time data '2025-13-08' does not match format '%Y-%m-%d'

# Solution: Use correct format
python scripts/step5_interpolation.py --date 2025-08-13  # YYYY-MM-DD
```

#### **Corrupted Data**
```bash
# Error: Cannot parse CSV
❌ ParserError: Error tokenizing data

# Solution: Check file integrity
# Re-download from S3
# Check for partial file downloads
```

## 🔧 Debugging Tools

### **Check Pipeline Status**
```bash
# Check what's been processed
find data/processed -name "*.csv" -exec ls -la {} \;

# Check logs for errors
tail -f logs/pipeline.log

# Check OSRM connectivity
curl "http://localhost:5005/nearest/v1/bike/0,0"
```

### **Validate Configuration**
```bash
# Test configuration loading
python -c "
from scripts.utils.config_manager import ConfigManager
config = ConfigManager()
config.validate_config()
print('Configuration OK')
"

# Test AWS connection
python -c "
from scripts.utils.aws_helper import AWSHelper
from scripts.utils.config_manager import ConfigManager
config = ConfigManager()
aws = AWSHelper(config.get_aws_config())
print('AWS OK' if aws.test_connection() else 'AWS Failed')
"
```

### **Manual Testing**
```bash
# Test individual steps
python scripts/step3_daily_combiner.py --date 2025/08/13
python scripts/step4_device_bifurcation.py --date 2025-08-13
python scripts/step5_interpolation.py --date 2025-08-13 --region england
python scripts/step6_combine_upload.py --date 2025-08-13

# Test with verbose logging
python scripts/main_pipeline.py --date 2025-08-13 --only-steps 5
```

## 📋 Health Check Checklist

Before reporting issues, verify:

- [ ] Docker is running and OSRM containers are active
- [ ] AWS credentials are valid and have S3 permissions
- [ ] Configuration files exist and are valid JSON
- [ ] Python environment has all required packages
- [ ] Directory structure exists with proper permissions
- [ ] Source data exists in S3 for the target date
- [ ] Device mapping includes all device prefixes
- [ ] OSRM servers respond to test queries
- [ ] No conflicting processes using the same directories

## 🆘 Getting Help

### **Collect Debug Information**
```bash
# System information
python --version
docker --version
conda --version

# Configuration check
./setup.sh 2>&1 | grep -E "(✅|❌|⚠️)"

# Recent logs
tail -50 logs/pipeline.log

# Directory status
find data -type f -name "*.csv" | head -10
```

### **Log Analysis**
```bash
# Find errors in logs
grep -i error logs/*.log

# Find warnings
grep -i warning logs/*.log

# Check specific step
grep "Step 5" logs/pipeline.log
```

### **Performance Issues**
```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check CPU usage
top -p $(pgrep -f "python.*pipeline")
```

## 🔄 Recovery Procedures

### **Restart from Specific Step**
```bash
# If Step 5 failed, restart from there
python scripts/main_pipeline.py --only-steps 5,6 --date 2025-08-13
```

### **Clean State Reset**
```bash
# Clear all processed data for a date
rm -rf data/preprocessed/*/2025-08-13
rm -rf data/processed/*/2025-08-13
rm -f data/combinedfile/combined_*20250813*.csv

# Restart pipeline
python scripts/main_pipeline.py --date 2025-08-13
```

### **Emergency Data Recovery**
```bash
# Download raw data manually
aws s3 sync s3://your-bucket/summit2/mqtt-flespi-barra/csv/2025/08/13/ data/downloadedfiles/

# Process locally without S3
python scripts/step4_device_bifurcation.py --date 2025-08-13
```