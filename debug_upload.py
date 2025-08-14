#!/usr/bin/env python3
"""
Debug script for S3 upload issue
Run this to identify the exact problem with the upload.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_manager import ConfigManager
from scripts.utils.aws_helper import AWSHelper

def debug_s3_upload():
    """Debug the S3 upload issue."""
    print("🔍 Debugging S3 Upload Issue")
    print("=" * 40)
    
    try:
        # Load configuration
        config = ConfigManager()
        aws_config = config.get_aws_config()
        
        print(f"✅ Configuration loaded")
        print(f"   - Bucket: {config.get('aws.bucket_name')}")
        print(f"   - Daily trips prefix: {config.get('aws.daily_trips_prefix')}")
        
        # Create AWS helper
        aws_helper = AWSHelper(aws_config)
        
        # Test AWS connection
        if not aws_helper.test_connection():
            print("❌ AWS connection failed")
            return False
        
        print("✅ AWS connection successful")
        
        # Test parameters
        date_str = "2025-08-10"
        date_parts = date_str.split('-')
        year, month, day = date_parts[0], date_parts[1], date_parts[2]
        
        destination_key = f"{config.get('aws.daily_trips_prefix')}year={year}/month={month}/day={day}/{year}{month}{day}_trips.csv"
        
        print(f"\n🎯 Upload Parameters:")
        print(f"   - Destination key: {destination_key}")
        print(f"   - Key type: {type(destination_key)}")
        print(f"   - Key length: {len(destination_key)}")
        
        # Check for a test file
        base_dir = Path(config.get('directories.base_dir', str(project_root)))
        combined_dir = base_dir / config.get('directories.combined_dir', 'data/combinedfile')
        
        # Look for any combined file
        combined_files = list(combined_dir.glob('combined_trips_*.csv'))
        
        if not combined_files:
            print("❌ No combined files found for testing")
            print(f"   - Checked directory: {combined_dir}")
            return False
        
        test_file = combined_files[0]
        test_file_str = str(test_file)
        
        print(f"\n📄 Test File:")
        print(f"   - File path: {test_file}")
        print(f"   - File path type: {type(test_file)}")
        print(f"   - File path string: {test_file_str}")
        print(f"   - File path string type: {type(test_file_str)}")
        print(f"   - File exists: {test_file.exists()}")
        print(f"   - File size: {test_file.stat().st_size if test_file.exists() else 'N/A'} bytes")
        
        # Test the upload with detailed error handling
        print(f"\n🚀 Testing Upload...")
        
        try:
            # Direct boto3 test
            print("Testing direct boto3 upload...")
            aws_helper.s3_client.upload_file(test_file_str, aws_helper.bucket_name, destination_key)
            print("✅ Direct boto3 upload successful")
            
        except Exception as e:
            print(f"❌ Direct boto3 upload failed: {e}")
            print(f"   - Error type: {type(e).__name__}")
            
            # Check each parameter
            print(f"\n🔍 Parameter Analysis:")
            print(f"   - test_file_str: '{test_file_str}' (type: {type(test_file_str)})")
            print(f"   - bucket_name: '{aws_helper.bucket_name}' (type: {type(aws_helper.bucket_name)})")
            print(f"   - destination_key: '{destination_key}' (type: {type(destination_key)})")
            
            # Check for None values
            if test_file_str is None:
                print("   ❌ test_file_str is None")
            if aws_helper.bucket_name is None:
                print("   ❌ bucket_name is None")
            if destination_key is None:
                print("   ❌ destination_key is None")
                
            return False
        
        # Test through helper
        print("\nTesting through AWS helper...")
        success = aws_helper.upload_file(test_file_str, destination_key)
        
        if success:
            print("✅ AWS helper upload successful")
        else:
            print("❌ AWS helper upload failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Debug script error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    debug_s3_upload()