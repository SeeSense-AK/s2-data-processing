# S2 Data Pipeline - Setup Guide

## 🚀 Quick Start

### Prerequisites
- macOS (tested on M1 Mac Mini)
- Docker Desktop installed and running
- Python 3.8+ or Conda
- AWS account with S3 access
- OSRM Docker containers running

### 1. Clone Repository
```bash
git clone https://github.com/YourOrg/s2-data-processing.git
cd s2-data-processing
```

### 2. Run Automated Setup
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate s2-data-processing
```

#### Option B: Using pip
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure AWS Credentials
```bash
# Copy template
cp config/aws_config_template.json config/aws_config.json

# Edit with your AWS credentials
nano config/aws_config.json
```

```json
{
  "aws_access_key_id": "YOUR_ACTUAL_ACCESS_KEY",
  "aws_secret_access_key": "YOUR_ACTUAL_SECRET_KEY",
  "region": "eu-west-1"
}
```

### 5. Configure Device Mapping
Edit `config/device_mapping.json` with your actual device IDs and regions.

### 6. Start OSRM Containers
```bash
# Start all OSRM containers
docker start osrm-england osrm-finland osrm-ireland osrm-sydney osrm-wales

# Verify they're running
docker ps | grep osrm
```

Expected output:
```
osrm-england    5005:5000
osrm-finland    5001:5000  
osrm-ireland    5002:5000
osrm-sydney     5003:5000
osrm-wales      5004:5000
```

### 7. Test Installation
```bash
# Test Step 3 setup
python tests/test_step3.py

# Test Step 5 setup  
python tests/test_step5.py
```

### 8. Run Pipeline
```bash
# Interactive mode
python scripts/main_pipeline.py

# Automated mode
python scripts/main_pipeline.py --automated

# Process specific date
python scripts/main_pipeline.py --date 2025-08-13
```

## 🔧 Detailed Configuration

### Pipeline Configuration (`config/pipeline_config.json`)
Key settings to review:
- `aws.bucket_name`: Your S3 bucket
- `directories.base_dir`: Pipeline working directory
- `processing.trip_break_threshold_minutes`: Trip segmentation threshold
- `osrm_servers`: OSRM container ports

### Device Mapping (`config/device_mapping.json`)
Map your device ID prefixes to regions:
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

## 📅 Scheduling (Production)

### Cron Job Setup
```bash
# Edit crontab
crontab -e

# Add daily execution at 3:30 AM
30 3 * * * cd /path/to/s2-data-processing && /path/to/conda/envs/s2-data-processing/bin/python scripts/main_pipeline.py --automated

# With logging
30 3 * * * cd /path/to/s2-data-processing && /path/to/conda/envs/s2-data-processing/bin/python scripts/main_pipeline.py --automated >> logs/cron.log 2>&1
```

### Systemd Service (Alternative)
Create `/etc/systemd/system/s2-pipeline.service`:
```ini
[Unit]
Description=S2 Data Pipeline
After=docker.service

[Service]
Type=oneshot
User=your-user
WorkingDirectory=/path/to/s2-data-processing
Environment=PATH=/path/to/conda/envs/s2-data-processing/bin
ExecStart=/path/to/conda/envs/s2-data-processing/bin/python scripts/main_pipeline.py --automated

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable s2-pipeline.service
sudo systemctl start s2-pipeline.service
```

## 🐳 OSRM Container Setup

If you need to create OSRM containers from scratch:

### Download Map Data
```bash
# Example for England
wget https://download.geofabrik.de/europe/great-britain/england-latest.osm.pbf
```

### Build and Run Container
```bash
# Prepare OSRM data
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-extract -p /opt/bike.lua /data/england-latest.osm.pbf
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-partition /data/england-latest.osrm
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-customize /data/england-latest.osrm

# Run OSRM server
docker run -t -i -p 5005:5000 -v "${PWD}:/data" --name osrm-england ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/england-latest.osrm
```

Repeat for each region with appropriate ports:
- England: 5005
- Finland: 5001  
- Ireland: 5002
- Sydney: 5003
- Wales: 5004

## 🔍 Verification Checklist

Before running the pipeline, ensure:

- [ ] All OSRM containers are running on correct ports
- [ ] AWS credentials are configured and valid
- [ ] S3 bucket access is working
- [ ] Device mapping matches your actual devices
- [ ] Python environment has all required packages
- [ ] Directory structure is created
- [ ] Tests pass successfully

## 🆘 Need Help?

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Review [CONFIGURATION.md](CONFIGURATION.md) for detailed settings
- Run tests to validate your setup
- Check logs in `logs/` directory for error details