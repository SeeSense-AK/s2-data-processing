#!/usr/bin/env python3
"""
Test Script for Step 3 - Daily CSV Combiner
Tests the configuration and AWS connectivity before running the main script.

Author: SeeSense Data Pipeline
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scripts.utils.config_manager import ConfigManager
from scripts.utils.aws_helper import AWSHelper
from scripts.utils.logger_setup import quick_logger


def test_configuration():
    """Test configuration loading."""
    print("🔧 Testing Configuration...")
    
    try:
        config = ConfigManager()
        config.validate_config()
        print("✅ Configuration loaded and validated successfully")
        
        # Print key configuration values (without sensitive data)
        print(f"   - Bucket: {config.get('aws.bucket_name')}")
        print(f"   - Source Prefix: {config.get('aws.source_prefix')}")
        print(f"   - Daily CSV Prefix: {config.get('aws.daily_csv_prefix')}")
        
        return config
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return None


def test_aws_connection(config):
    """Test AWS S3 connectivity."""
    print("\n🌐 Testing AWS S3 Connection...")
    
    try:
        aws_helper = AWSHelper(config.get_aws_config())
        
        if aws_helper.test_connection():
            print("✅ AWS S3 connection successful")
            
            # Test bucket access
            bucket_info = aws_helper.get_bucket_info()
            if bucket_info and bucket_info.get('exists'):
                print(f"✅ Bucket access confirmed: {bucket_info['name']}")
                return aws_helper
            else:
                print("❌ Cannot access the specified bucket")
                return None
        else:
            print("❌ AWS S3 connection failed")
            return None
            
    except Exception as e:
        print(f"❌ AWS connection error: {e}")
        return None


def check_available_dates(aws_helper, config):
    """Check what dates have data available."""
    print("\n📅 Checking Available Dates...")
    
    try:
        bucket = config.get('aws.bucket_name')
        source_prefix = config.get('aws.source_prefix')
        
        # Check for recent dates
        today = datetime.utcnow().strftime('%Y/%m/%d')
        yesterday = (datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) 
                    - datetime.timedelta(days=1)).strftime('%Y/%m/%d')
        
        print(f"Checking for data on:")
        print(f"  - Today: {today}")
        print(f"  - Yesterday: {yesterday}")
        
        available_dates = []
        
        for date_str in [today, yesterday]:
            prefix = f"{source_prefix}{date_str}/"
            files = aws_helper.list_files(prefix)
            if files:
                available_dates.append(date_str)
                print(f"✅ {date_str}: {len(files)} files found")
            else:
                print(f"❌ {date_str}: No files found")
        
        if available_dates:
            print(f"\n📊 Recommended test date: {available_dates[0]}")
            return available_dates[0]
        else:
            print("\n❌ No data found for recent dates")
            
            # Try to find any available dates
            print("🔍 Searching for any available data...")
            
            # Check the current month
            current_year_month = datetime.utcnow().strftime('%Y/%m')
            month_prefix = f"{source_prefix}{current_year_month}/"
            month_files = aws_helper.list_files(month_prefix)
            
            if month_files:
                # Extract unique dates
                dates = set()
                for file in month_files[:10]:  # Check first 10 files
                    parts = file.replace(source_prefix, '').split('/')
                    if len(parts) >= 3:
                        dates.add(f"{parts[0]}/{parts[1]}/{parts[2]}")
                
                if dates:
                    sorted_dates = sorted(dates, reverse=True)
                    print(f"📅 Found data for dates: {sorted_dates[:3]}")  # Show first 3
                    return sorted_dates[0]
            
            return None
            
    except Exception as e:
        print(f"❌ Error checking dates: {e}")
        return None


def test_step3_dry_run(config, aws_helper, test_date):
    """Perform a dry run test of Step 3."""
    print(f"\n🧪 Testing Step 3 (Dry Run) for date: {test_date}")
    
    try:
        bucket = config.get('aws.bucket_name')
        source_prefix = f"{config.get('aws.source_prefix')}{test_date}/"
        
        # List files
        print(f"📂 Listing files in: s3://{bucket}/{source_prefix}")
        files = aws_helper.list_files(source_prefix)
        
        if files:
            print(f"✅ Found {len(files)} files to process")
            print("📋 Sample files:")
            for i, file in enumerate(files[:3]):  # Show first 3
                file_info = aws_helper.get_file_info(file)
                size_mb = file_info['size'] / (1024*1024) if file_info else 0
                print(f"   {i+1}. {file.split('/')[-1]} ({size_mb:.2f} MB)")
            
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more files")
            
            # Check destination path
            destination_key = f"{config.get('aws.daily_csv_prefix')}year={test_date.split('/')[0]}/month={test_date.split('/')[1]}/day={test_date.split('/')[2]}/{test_date.replace('/', '')}.csv"
            print(f"\n📤 Would upload to: s3://{bucket}/{destination_key}")
            
            return True
        else:
            print(f"❌ No files found for {test_date}")
            return False
            
    except Exception as e:
        print(f"❌ Step 3 dry run error: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 S2 Data Pipeline - Step 3 Test Suite")
    print("=" * 50)
    
    # Test 1: Configuration
    config = test_configuration()
    if not config:
        print("\n❌ Configuration test failed. Please check your config files.")
        sys.exit(1)
    
    # Test 2: AWS Connection
    aws_helper = test_aws_connection(config)
    if not aws_helper:
        print("\n❌ AWS connection test failed. Please check your credentials.")
        sys.exit(1)
    
    # Test 3: Check Available Data
    test_date = check_available_dates(aws_helper, config)
    if not test_date:
        print("\n❌ No test data available. Please check your S3 bucket.")
        sys.exit(1)
    
    # Test 4: Dry Run
    if test_step3_dry_run(config, aws_helper, test_date):
        print("\n✅ All tests passed!")
        print("\n🎯 Ready to run Step 3:")
        print(f"   python scripts/step3_daily_combiner.py --date {test_date}")
        print("   OR")
        print("   python scripts/step3_daily_combiner.py  # for interactive mode")
    else:
        print("\n❌ Dry run failed. Check the logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()