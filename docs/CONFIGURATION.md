# S2 Data Pipeline - Configuration Guide

## 📋 Configuration Files Overview

The pipeline uses three main configuration files:

1. `config/pipeline_config.json` - Main pipeline settings
2. `config/device_mapping.json` - Device-to-region mapping  
3. `config/aws_config.json` - AWS credentials (not in Git)

## 🔧 Pipeline Configuration (`pipeline_config.json`)

### Complete Configuration Example
```json
{
  "aws": {
    "bucket_name": "seesense-air",
    "source_prefix": "summit2/mqtt-flespi-barra/csv/",
    "daily_csv_prefix": "summit2/mqtt-flespi-barra/dailycsv/",
    "daily_trips_prefix": "summit2/mqtt-flespi-barra/dailytripscsv/",
    "region": "eu-west-1"
  },
  "osrm_servers": {
    "finland": {
      "host": "localhost",
      "port": 5001,
      "container_name": "osrm-finland"
    },
    "ireland": {
      "host": "localhost", 
      "port": 5002,
      "container_name": "osrm-ireland"
    },
    "sydney": {
      "host": "localhost",
      "port": 5003,
      "container_name": "osrm-sydney"
    },
    "wales": {
      "host": "localhost",
      "port": 5004,
      "container_name": "osrm-wales"
    },
    "england": {
      "host": "localhost",
      "port": 5005,
      "container_name": "osrm-england"
    }
  },
  "directories": {
    "base_dir": "/Users/abhishekkumbhar/Documents/s2-data-processing",
    "download_dir": "data/downloadedfiles",
    "combined_dir": "data/combinedfile",
    "preprocessed_dir": "data/preprocessed",
    "processed_dir": "data/processed",
    "logs_dir": "logs"
  },
  "processing": {
    "retention_days": 2,
    "max_workers": 10,
    "chunk_size": 1000,
    "trip_break_threshold_minutes": 30,
    "max_speed_threshold_kmh": 200
  },
  "schedule": {
    "daily_run_time": "03:30",
    "timezone": "UTC"
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "max_file_size_mb": 10,
    "backup_count": 5
  }
}
```

### Configuration Sections

#### **AWS Settings**
```json
"aws": {
  "bucket_name": "your-s3-bucket",           // S3 bucket name
  "source_prefix": "path/to/csv/files/",    // Source CSV location
  "daily_csv_prefix": "path/to/daily/",     // Combined CSV output
  "daily_trips_prefix": "path/to/trips/",   // Final trips output
  "region": "eu-west-1"                     // AWS region
}
```

#### **OSRM Servers**
```json
"osrm_servers": {
  "region_name": {
    "host": "localhost",                     // OSRM server host
    "port": 5001,                          // OSRM server port
    "container_name": "osrm-region"        // Docker container name
  }
}
```

**Port Mapping:**
- Finland: 5001
- Ireland: 5002  
- Sydney: 5003
- Wales: 5004
- England: 5005

#### **Directories**
```json
"directories": {
  "base_dir": "/full/path/to/project",      // Absolute project path
  "download_dir": "data/downloadedfiles",   // Temp CSV downloads
  "combined_dir": "data/combinedfile",      // Daily combined files
  "preprocessed_dir": "data/preprocessed",  // Regional split data
  "processed_dir": "data/processed",        // After interpolation
  "logs_dir": "logs"                        // Log files
}
```

#### **Processing Settings**
```json
"processing": {
  "retention_days": 2,                      // Local data retention
  "max_workers": 10,                       // Concurrent downloads
  "chunk_size": 1000,                     // Processing batch size
  "trip_break_threshold_minutes": 30,      // Trip segmentation
  "max_speed_threshold_kmh": 200           // Speed filter
}
```

#### **Logging**
```json
"logging": {
  "level": "INFO",                         // DEBUG, INFO, WARNING, ERROR
  "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  "max_file_size_mb": 10,                 // Log rotation size
  "backup_count": 5                       // Number of backup logs
}
```

## 🗺️ Device Mapping (`device_mapping.json`)

### Configuration Format
```json
{
  "device_prefixes": {
    "england": ["AS", "BAE", "BR", "KEE", "MK", "SBY", "AB", "AD", "AC", "HL", "AH", "GH"],
    "sydney": ["TEBT"],
    "finland": ["HEL"], 
    "ireland": ["SPIN", "XX"],
    "wales": ["SW"]
  },
  "region_info": {
    "england": {
      "country": "United Kingdom",
      "timezone": "Europe/London",
      "osrm_port": 5005
    },
    "sydney": {
      "country": "Australia", 
      "timezone": "Australia/Sydney",
      "osrm_port": 5003
    },
    "finland": {
      "country": "Finland",
      "timezone": "Europe/Helsinki", 
      "osrm_port": 5001
    },
    "ireland": {
      "country": "Ireland",
      "timezone": "Europe/Dublin",
      "osrm_port": 5002
    },
    "wales": {
      "country": "United Kingdom",
      "timezone": "Europe/London",
      "osrm_port": 5004
    }
  }
}
```

### Adding New Devices
1. Identify device ID prefix (e.g., "LON" for London devices)
2. Add to appropriate region in `device_prefixes`
3. Update if new region is needed

### Adding New Regions
1. Add region to `device_prefixes`
2. Add region info with timezone and OSRM port
3. Set up corresponding OSRM container
4. Update pipeline configuration

## 🔐 AWS Configuration (`aws_config.json`)

**⚠️ NEVER commit this file to Git!**

### Configuration Format
```json
{
  "aws_access_key_id": "AKIA...",
  "aws_secret_access_key": "...",
  "region": "eu-west-1"
}
```

### Creating AWS Credentials
1. AWS Console → IAM → Users → Create User
2. Attach policies: `AmazonS3FullAccess` (or custom S3 policy)
3. Create access key → Download credentials
4. Add to `aws_config.json`

### Custom S3 Policy (Recommended)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject", 
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

## 🛠️ Environment-Specific Configurations

### Development Environment
```json
{
  "aws": {
    "bucket_name": "seesense-air-dev",
    "daily_csv_prefix": "dev/dailycsv/",
    "daily_trips_prefix": "dev/dailytripscsv/"
  },
  "processing": {
    "retention_days": 1,
    "max_workers": 5
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

### Production Environment  
```json
{
  "aws": {
    "bucket_name": "seesense-air",
    "daily_csv_prefix": "summit2/mqtt-flespi-barra/dailycsv/",
    "daily_trips_prefix": "summit2/mqtt-flespi-barra/dailytripscsv/"
  },
  "processing": {
    "retention_days": 2,
    "max_workers": 10
  },
  "logging": {
    "level": "INFO"
  }
}
```

## 🔄 Configuration Updates

### Updating Without Restart
Most configuration changes require pipeline restart. However, some settings are read dynamically:
- Log levels (take effect on next step)
- Processing thresholds (next file processing)

### Configuration Validation
```bash
# Test configuration
python -c "
from scripts.utils.config_manager import ConfigManager
config = ConfigManager()
config.validate_config()
print('✅ Configuration valid')
"
```

### Backup Configurations
```bash
# Backup current config before changes
cp config/pipeline_config.json config/pipeline_config.json.backup.$(date +%Y%m%d)
```

## 📊 Performance Tuning

### High-Volume Processing
```json
"processing": {
  "max_workers": 20,           // More concurrent downloads
  "chunk_size": 5000,         // Larger processing batches
  "retention_days": 1         // Less local storage
}
```

### Resource-Constrained Environment
```json
"processing": {
  "max_workers": 5,            // Fewer concurrent operations
  "chunk_size": 500,          // Smaller batches
  "retention_days": 7         // More local caching
}
```

### Network-Optimized
```json
"processing": {
  "max_workers": 15,          // Balance concurrency
  "trip_break_threshold_minutes": 60,  // Larger trips
  "max_speed_threshold_kmh": 300       // Less filtering
}
```

## 🔧 Advanced Settings

### Custom S3 Paths
Use date-based partitioning:
```json
"aws": {
  "source_prefix": "data/csv/year={year}/month={month}/day={day}/",
  "daily_csv_prefix": "processed/daily/year={year}/month={month}/",
  "daily_trips_prefix": "analytics/trips/year={year}/month={month}/"
}
```

### Multi-Region OSRM
For distributed OSRM servers:
```json
"osrm_servers": {
  "england": {
    "host": "osrm-england.example.com",
    "port": 443,
    "ssl": true
  }
}
```

### Custom Processing Rules
```json
"processing": {
  "filters": {
    "min_trip_duration_seconds": 60,
    "min_trip_distance_meters": 100,
    "max_gap_seconds": 300
  }
}
```