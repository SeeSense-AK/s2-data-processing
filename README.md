# S2 Data Processing Pipeline

A comprehensive data processing pipeline for SeeSense S2 device data, featuring automated CSV processing, device bifurcation, OSRM-based interpolation, and AWS S3 integration.

## Overview

This pipeline processes MQTT streaming data from Flespi through several stages:

1. **Daily CSV Combination** - Combines individual CSV files into daily aggregates
2. **Device Bifurcation** - Sorts devices by region using mapping dictionary  
3. **OSRM Interpolation** - Adds time and distance calculations between data points
4. **Regional Combination** - Combines processed regional data and uploads to S3

## Architecture

```
AWS S3 (CSV Files) → Daily Combiner → Device Bifurcation → OSRM Interpolation → Regional Combiner → AWS S3 (Processed)
```

## OSRM Servers

The pipeline uses region-specific OSRM containers running locally:

- **Finland**: `localhost:5001` (osrm-finland)
- **Ireland**: `localhost:5002` (osrm-ireland) 
- **Sydney**: `localhost:5003` (osrm-sydney)
- **Wales**: `localhost:5004` (osrm-wales)
- **England**: `localhost:5005` (osrm-england)

## Setup

### Prerequisites

- macOS (tested on M1 Mac Mini)
- Docker with OSRM containers running
- Conda package manager
- AWS credentials with S3 access

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SeeSense-AK/s2-data-processing.git
   cd s2-data-processing
   ```

2. **Create and activate conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate s2-data-processing
   ```

3. **Create directory structure:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

4. **Configure AWS credentials:**
   ```bash
   cp config/aws_config_template.json config/aws_config.json
   # Edit aws_config.json with your actual AWS credentials
   ```

5. **Update device mapping:**
   ```bash
   # Edit config/device_mapping.json with your actual device IDs and regions
   ```

## Configuration

### Pipeline Configuration (`config/pipeline_config.json`)
- AWS S3 bucket settings
- OSRM server endpoints  
- Processing parameters
- Directory paths
- Scheduling settings

### Device Mapping (`config/device_mapping.json`)
- Device ID to region mapping
- Region-specific settings
- Timezone configurations

## Usage

### Manual Execution

```bash
# Activate environment
conda activate s2-data-processing

# Run complete pipeline
python scripts/main_pipeline.py

# Run individual steps
python scripts/step3_daily_combiner.py
python scripts/step5_device_bifurcation.py
python scripts/step6_interpolation.py  
python scripts/step7_combine_upload.py
```

### Scheduled Execution

The pipeline is designed to run daily at 3:30 AM via cron:

```bash
# Add to crontab
30 3 * * * /path/to/conda/envs/s2-data-processing/bin/python /path/to/s2-data-processing/scripts/main_pipeline.py
```

## Directory Structure

```
data/
├── downloadedfiles/     # Temporary CSV downloads from S3
├── combinedfile/        # Daily combined CSV files  
├── preprocessed/        # Region-wise bifurcated data (2-day retention)
│   ├── finland/
│   ├── ireland/
│   ├── sydney/
│   ├── wales/
│   └── england/
└── processed/          # After OSRM interpolation
    ├── finland/
    ├── ireland/
    ├── sydney/
    ├── wales/
    └── england/
```

## Data Flow

1. **Input**: Individual CSV files from S3 (`summit2/mqtt-flespi-barra/csv/YYYY/MM/DD/`)
2. **Step 3**: Download and combine into single daily CSV
3. **Step 5**: Split by device regions into separate CSV files
4. **Step 6**: Process each region through OSRM for interpolation
5. **Step 7**: Combine all processed regions and upload to S3 (`summit2/mqtt-flespi-barra/dailytripscsv/`)

## Monitoring

- Logs stored in `logs/` directory
- Configurable log levels and rotation
- Error notifications and retry mechanisms

## Development

### Testing
```bash
python -m pytest tests/
```

### Adding New Regions
1. Update `config/device_mapping.json`
2. Add new OSRM container configuration
3. Update region lists in processing scripts

## Troubleshooting

### Common Issues

1. **OSRM Connection Failed**: Verify Docker containers are running
2. **AWS Permission Denied**: Check S3 bucket permissions
3. **Memory Issues**: Adjust chunk_size in configuration

### Logs Location
Check `logs/pipeline.log` for detailed execution logs.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes and test
4. Submit pull request

## License

Internal SeeSense project - All rights reserved.
