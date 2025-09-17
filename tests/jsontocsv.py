#!/usr/bin/env python3
"""
JSON to CSV S3 Processor with Fixed Date Structure
=================================================

This script replaces the Lambda function locally:
1. Downloads JSON files from S3 (with dot notation fields)
2. Converts field names from dots to underscores
3. Converts to CSV format with underscore headers
4. Uploads CSV files back to S3 MAINTAINING THE SAME DATE STRUCTURE

Fixed: Uses original file date structure instead of current date
Example: JSON at 2025/07/02/ -> CSV at 2025/07/02/ (not today's date)
"""

import boto3
import csv
import json
import os
from pathlib import Path
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import tempfile

# ============ CONFIGURATION ============
SOURCE_S3_BUCKET = "seesense-air"
SOURCE_S3_PREFIX = "summit2/mqtt-flespi-barra/flespi-replay/"  # 🔥 UPDATE THIS PATH
DEST_S3_BUCKET = "seesense-air"
DEST_S3_PREFIX = "summit2/mqtt-flespi-barra/flespi-replay/csv/"

# Processing Configuration
MAX_WORKERS = 8  # Number of parallel processing threads
BATCH_SIZE = 100  # Number of files to process per batch
DEBUG_MODE = False  # Set to True for detailed logging

# Fields to extract from JSON (with dot notation - as they appear in your JSON files)
FIELDS_TO_CHECK_DOTS = [
    'battery.voltage', 'device.id', 'device.name', 'device.serial.number', 'device.temperature', 
    'gsm.signal.quality', 'ident', 'loaded.battery.voltage', 'position.accuracy', 'position.altitude', 
    'position.direction', 'position.latitude', 'position.longitude', 'position.pdop', 'position.speed', 
    'position.timestamp', 'record.seqnum', 'report.reason', 'rtc.timestamp', 'server.timestamp', 
    'timestamp', 'trip.status'
]

# CSV column headers (with underscores - as they should appear in CSV)
CSV_HEADERS = [
    'battery_voltage', 'device_id', 'device_name', 'device_serial_number', 'device_temperature', 
    'gsm_signal_quality', 'ident', 'loaded_battery_voltage', 'position_accuracy', 'position_altitude', 
    'position_direction', 'position_latitude', 'position_longitude', 'position_pdop', 'position_speed', 
    'position_timestamp', 'record_seqnum', 'report_reason', 'rtc_timestamp', 'server_timestamp', 
    'timestamp', 'trip_status'
]

# Thread-safe progress tracking
process_lock = threading.Lock()
process_stats = {
    'files_processed': 0,
    'files_failed': 0,
    'csv_files_created': 0,
    'start_time': None
}

print("🔄 JSON to CSV Converter - FIXED DATE STRUCTURE")
print("=" * 70)
print(f"📂 Source: s3://{SOURCE_S3_BUCKET}/{SOURCE_S3_PREFIX}")
print(f"📁 Destination: s3://{DEST_S3_BUCKET}/{DEST_S3_PREFIX}")
print(f"🔄 Field conversion: battery.voltage -> battery_voltage")
print(f"📅 Date structure: MAINTAINS original file dates (not current date)")
print(f"⚡ Parallel Workers: {MAX_WORKERS}")

def initialize_s3_client():
    """Initialize and test S3 connection"""
    print("\n🔧 Initializing S3 connection...")
    
    try:
        s3_client = boto3.client('s3')
        
        # Test connection
        s3_client.head_bucket(Bucket=SOURCE_S3_BUCKET)
        print(f"✅ S3 connection successful")
        
        return s3_client
        
    except NoCredentialsError:
        print("❌ AWS credentials not found")
        print("   Please configure AWS CLI or set environment variables")
        return None
    except ClientError as e:
        print(f"❌ S3 connection failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected S3 error: {e}")
        return None

def show_field_mapping():
    """Display the field name conversion mapping"""
    print(f"\n📋 FIELD NAME CONVERSION MAPPING:")
    print("-" * 50)
    print(f"{'JSON Field (dots)': <25} -> {'CSV Header (underscores)': <25}")
    print("-" * 50)
    
    for i in range(min(10, len(FIELDS_TO_CHECK_DOTS))):  # Show first 10 mappings
        json_field = FIELDS_TO_CHECK_DOTS[i]
        csv_field = CSV_HEADERS[i]
        print(f"{json_field: <25} -> {csv_field: <25}")
    
    if len(FIELDS_TO_CHECK_DOTS) > 10:
        print(f"... and {len(FIELDS_TO_CHECK_DOTS) - 10} more field mappings")
    
    print(f"\n📊 Total fields to extract: {len(FIELDS_TO_CHECK_DOTS)}")

def get_json_files_from_s3(s3_client, date_filter=None):
    """Get list of JSON files from S3 source prefix"""
    
    print(f"\n📂 Scanning S3 for JSON files...")
    
    json_files = []
    continuation_token = None
    
    try:
        while True:
            # Prepare list_objects_v2 parameters
            list_params = {
                'Bucket': SOURCE_S3_BUCKET,
                'Prefix': SOURCE_S3_PREFIX,
                'MaxKeys': 1000
            }
            
            if continuation_token:
                list_params['ContinuationToken'] = continuation_token
            
            response = s3_client.list_objects_v2(**list_params)
            
            if 'Contents' not in response:
                break
            
            for obj in response['Contents']:
                key = obj['Key']
                
                # Only process JSON files
                if key.endswith('.json'):
                    # Apply date filter if provided
                    if date_filter:
                        # Extract date from S3 key path (assuming YYYY/MM/DD structure)
                        parts = key.split('/')
                        if len(parts) >= 4:
                            try:
                                file_date = f"{parts[-4]}-{parts[-3]}-{parts[-2]}"  # YYYY-MM-DD
                                if file_date != date_filter:
                                    continue
                            except (IndexError, ValueError):
                                # If we can't parse date from path, include the file
                                pass
                    
                    json_files.append({
                        'key': key,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
            
            # Check if there are more objects to retrieve
            if not response.get('IsTruncated', False):
                break
            
            continuation_token = response.get('NextContinuationToken')
    
    except Exception as e:
        print(f"❌ Error scanning S3: {e}")
        return []
    
    print(f"✅ Found {len(json_files)} JSON files to process")
    
    if json_files:
        total_size_mb = sum(f['size'] for f in json_files) / (1024 * 1024)
        print(f"📊 Total size: {total_size_mb:.2f} MB")
        
        # Show sample files
        print(f"📝 Sample files:")
        for file_info in json_files[:3]:
            print(f"   - {file_info['key'].split('/')[-1]} ({file_info['size']} bytes)")
    
    return json_files

def process_json_to_csv(json_file_info, s3_client, worker_id):
    """Download JSON file, convert field names and create CSV, upload to S3"""
    
    key = json_file_info['key']
    
    if DEBUG_MODE:
        print(f"[W{worker_id}] Processing: {key}")
    
    try:
        # Step 1: Download JSON file from S3
        response = s3_client.get_object(Bucket=SOURCE_S3_BUCKET, Key=key)
        json_content = response['Body'].read().decode('utf-8')
        
        if DEBUG_MODE:
            print(f"[W{worker_id}] Downloaded JSON content from {key}")
        
        # Step 2: Parse JSON
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"⚠️ [W{worker_id}] JSON decode error for {key}: {e}")
            return False, "JSON decode error"
        
        # Step 3: Extract and convert field names from dots to underscores
        filtered_data = {}
        
        for i, dot_field in enumerate(FIELDS_TO_CHECK_DOTS):
            csv_field = CSV_HEADERS[i]  # Corresponding underscore field name
            
            # Get value from JSON (with dot notation)
            value = data.get(dot_field, None)
            
            # Store with underscore notation for CSV
            filtered_data[csv_field] = value
        
        if DEBUG_MODE:
            print(f"[W{worker_id}] Field conversions:")
            for i in range(min(5, len(FIELDS_TO_CHECK_DOTS))):
                dot_field = FIELDS_TO_CHECK_DOTS[i]
                csv_field = CSV_HEADERS[i]
                value = filtered_data[csv_field]
                print(f"  {dot_field} -> {csv_field} = {value}")
        
        # Step 4: Generate CSV content
        csv_content = ""
        with tempfile.NamedTemporaryFile(mode='w', newline='', delete=False, suffix='.csv') as temp_csv:
            writer = csv.DictWriter(temp_csv, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerow(filtered_data)
            temp_csv_path = temp_csv.name
        
        # Read the CSV content
        with open(temp_csv_path, 'r') as csvfile:
            csv_content = csvfile.read()
        
        # Clean up temp file
        os.unlink(temp_csv_path)
        
        # Step 5: Generate CSV S3 key based on ORIGINAL file date (not current date)
        # Extract date from the original JSON file path
        file_parts = key.split('/')
        
        # Try to extract date from S3 path structure (YYYY/MM/DD)
        if len(file_parts) >= 4:
            try:
                # Assuming structure: .../YYYY/MM/DD/filename.json
                file_year = file_parts[-4]
                file_month = file_parts[-3] 
                file_day = file_parts[-2]
                
                # Validate it's actually a date format
                datetime.strptime(f"{file_year}-{file_month}-{file_day}", "%Y-%m-%d")
                
                if DEBUG_MODE:
                    print(f"[W{worker_id}] Extracted date from path: {file_year}/{file_month}/{file_day}")
                
            except (ValueError, IndexError):
                # Fallback to current date if we can't parse the path
                now = datetime.utcnow()
                file_year = now.strftime("%Y")
                file_month = now.strftime("%m")
                file_day = now.strftime("%d")
                
                if DEBUG_MODE:
                    print(f"[W{worker_id}] Could not parse date from path, using current date")
        else:
            # Fallback to current date if path structure is unexpected
            now = datetime.utcnow()
            file_year = now.strftime("%Y")
            file_month = now.strftime("%m")
            file_day = now.strftime("%d")
            
            if DEBUG_MODE:
                print(f"[W{worker_id}] Unexpected path structure, using current date")
        
        # Get original filename without extension
        original_filename = key.split('/')[-1]
        csv_filename = original_filename.rsplit('.', 1)[0] + '.csv'
        csv_key = f"{DEST_S3_PREFIX}{file_year}/{file_month}/{file_day}/{csv_filename}"
        
        # Step 6: Upload CSV to S3
        s3_client.put_object(
            Bucket=DEST_S3_BUCKET,
            Key=csv_key,
            Body=csv_content,
            ContentType='text/csv'
        )
        
        # Update progress
        with process_lock:
            process_stats['files_processed'] += 1
            process_stats['csv_files_created'] += 1
            
            if process_stats['files_processed'] % 100 == 0:
                print(f"📊 Progress: {process_stats['files_processed']} files processed...")
        
        if DEBUG_MODE:
            print(f"[W{worker_id}] ✅ Created CSV: {csv_key}")
        
        return True, csv_key
        
    except ClientError as e:
        error_msg = f"S3 error: {e}"
        if DEBUG_MODE:
            print(f"[W{worker_id}] ❌ {error_msg}")
        
        with process_lock:
            process_stats['files_failed'] += 1
        
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        if DEBUG_MODE:
            print(f"[W{worker_id}] ❌ {error_msg}")
        
        with process_lock:
            process_stats['files_failed'] += 1
        
        return False, error_msg

def process_files_parallel(json_files, s3_client):
    """Process all JSON files in parallel"""
    
    if not json_files:
        print("⚠️ No files to process")
        return
    
    total_files = len(json_files)
    
    print(f"\n🚀 Starting parallel processing...")
    print(f"📊 Total files: {total_files:,}")
    print(f"⚡ Using {MAX_WORKERS} parallel workers")
    print(f"🔄 Converting dots to underscores in field names")
    print(f"📅 Maintaining original file date structure")
    print("-" * 70)
    
    process_stats['start_time'] = datetime.now()
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_json_to_csv, file_info, s3_client, i % MAX_WORKERS + 1): file_info 
            for i, file_info in enumerate(json_files)
        }
        
        # Collect results as they complete
        successful_files = []
        failed_files = []
        
        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            try:
                success, result = future.result()
                if success:
                    successful_files.append((file_info['key'], result))
                else:
                    failed_files.append((file_info['key'], result))
            except Exception as e:
                failed_files.append((file_info['key'], f"Worker exception: {e}"))
                with process_lock:
                    process_stats['files_failed'] += 1
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - process_stats['start_time']
    
    files_per_second = process_stats['files_processed'] / duration.total_seconds() if duration.total_seconds() > 0 else 0
    
    print(f"\n✅ PROCESSING COMPLETE")
    print("=" * 70)
    print(f"⏱️  Duration: {duration}")
    print(f"📊 Files processed: {process_stats['files_processed']:,}")
    print(f"📄 CSV files created: {process_stats['csv_files_created']:,}")
    print(f"❌ Files failed: {process_stats['files_failed']:,}")
    print(f"🚀 Processing speed: {files_per_second:.1f} files/second")
    print(f"🔄 Field conversions: dots -> underscores completed")
    print(f"📅 Date structure: Original file dates maintained")
    
    # Show failed files if any
    if failed_files:
        print(f"\n❌ FAILED FILES ({len(failed_files)}):")
        for file_key, error in failed_files[:10]:  # Show first 10
            print(f"   - {file_key.split('/')[-1]}: {error}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more")
    
    # Show sample successful conversions
    if successful_files:
        print(f"\n✅ SAMPLE SUCCESSFUL CONVERSIONS:")
        for file_key, csv_key in successful_files[:3]:
            json_filename = file_key.split('/')[-1]
            csv_filename = csv_key.split('/')[-1]
            # Show the date structure preservation
            json_date_part = '/'.join(file_key.split('/')[-4:-1])  # YYYY/MM/DD
            csv_date_part = '/'.join(csv_key.split('/')[-4:-1])    # YYYY/MM/DD
            print(f"   - {json_date_part}/{json_filename} -> {csv_date_part}/{csv_filename}")

def verify_csv_upload(s3_client, date_filter=None):
    """Verify CSV files were uploaded to S3 and show sample content"""
    print(f"\n🔍 Verifying CSV upload...")
    
    try:
        # Use the date filter if provided, otherwise use current date
        if date_filter:
            # Convert YYYY-MM-DD to YYYY/MM/DD for S3 path
            date_parts = date_filter.split('-')
            verification_prefix = f"{DEST_S3_PREFIX}{date_parts[0]}/{date_parts[1]}/{date_parts[2]}/"
            print(f"🗓️ Checking for CSV files created for date: {date_filter}")
        else:
            # Fallback to current date
            now = datetime.utcnow()
            verification_prefix = f"{DEST_S3_PREFIX}{now.strftime('%Y/%m/%d')}/"
            print(f"🗓️ Checking for CSV files created today")
        
        response = s3_client.list_objects_v2(
            Bucket=DEST_S3_BUCKET,
            Prefix=verification_prefix,
            MaxKeys=100
        )
        
        if 'Contents' in response:
            csv_count = len(response['Contents'])
            total_size = sum(obj['Size'] for obj in response['Contents'])
            
            print(f"✅ Found {csv_count} CSV files")
            print(f"📊 Total CSV size: {total_size / 1024:.2f} KB")
            print(f"📁 Location: s3://{DEST_S3_BUCKET}/{verification_prefix}")
            
            # Show sample CSV files
            print(f"📝 Sample CSV files:")
            for obj in response['Contents'][:3]:
                print(f"   - {obj['Key'].split('/')[-1]}")
            
            # Try to read and show a sample CSV content
            if response['Contents']:
                sample_key = response['Contents'][0]['Key']
                try:
                    csv_response = s3_client.get_object(Bucket=DEST_S3_BUCKET, Key=sample_key)
                    csv_content = csv_response['Body'].read().decode('utf-8')
                    
                    print(f"\n📄 SAMPLE CSV CONTENT ({sample_key.split('/')[-1]}):")
                    print("-" * 50)
                    lines = csv_content.strip().split('\n')
                    for line in lines[:3]:  # Show header + first 2 rows
                        print(f"   {line}")
                    if len(lines) > 3:
                        print(f"   ... (truncated)")
                        
                except Exception as e:
                    print(f"⚠️ Could not read sample CSV: {e}")
            
            return csv_count
        else:
            print(f"⚠️ No CSV files found at: s3://{DEST_S3_BUCKET}/{verification_prefix}")
            return 0
            
    except Exception as e:
        print(f"❌ CSV verification error: {e}")
        return 0

def main():
    """Main execution function"""
    
    print(f"🔧 CONFIGURATION:")
    print(f"   📂 Source: s3://{SOURCE_S3_BUCKET}/{SOURCE_S3_PREFIX}")
    print(f"   📁 Destination: s3://{DEST_S3_BUCKET}/{DEST_S3_PREFIX}")
    print(f"   📋 Fields extracted: {len(FIELDS_TO_CHECK_DOTS)} fields")
    print(f"   🔄 Field conversion: dots -> underscores")
    print(f"   📅 Date structure: MAINTAINS original file dates")
    print(f"   ⚡ Workers: {MAX_WORKERS}")
    
    # Step 1: Show field mapping
    show_field_mapping()
    
    # Step 2: Initialize S3
    s3_client = initialize_s3_client()
    if s3_client is None:
        print("❌ Cannot proceed without S3 connection")
        return
    
    # Step 3: Get JSON files to process
    date_filter = input("\n📅 Enter specific date to process (YYYY-MM-DD) or press Enter for all: ").strip()
    if not date_filter:
        date_filter = None
    
    json_files = get_json_files_from_s3(s3_client, date_filter)
    
    if not json_files:
        print("❌ No JSON files found to process")
        return
    
    # Step 4: Show what will be processed
    total_size_mb = sum(f['size'] for f in json_files) / (1024 * 1024)
    
    print(f"\n⚠️  PROCESSING CONFIRMATION")
    print(f"📊 JSON files to process: {len(json_files):,}")
    print(f"📊 Total size: {total_size_mb:.2f} MB")
    print(f"📄 Will create {len(json_files):,} CSV files")
    print(f"🔄 Field conversion: battery.voltage -> battery_voltage (and {len(FIELDS_TO_CHECK_DOTS)-1} more)")
    print(f"📅 Date preservation: CSV files will use SAME dates as JSON files")
    
    if date_filter:
        print(f"📅 Processing files for date: {date_filter}")
        print(f"📁 CSV files will be saved to: csv/{date_filter.replace('-', '/')}/")
    else:
        print(f"📅 Processing ALL files in source")
        print(f"📁 CSV files will maintain their original date structure")
    
    confirm = input(f"\n❓ Proceed with JSON to CSV conversion? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("🚫 Processing cancelled")
        return
    
    # Step 5: Process files
    process_files_parallel(json_files, s3_client)
    
    # Step 6: Verify results
    verify_csv_upload(s3_client, date_filter)
    
    print(f"\n🎉 Processing complete!")
    print(f"   📄 CSV files with underscore headers available at:")
    print(f"   s3://{DEST_S3_BUCKET}/{DEST_S3_PREFIX}")
    
    print(f"\n📋 PROCESSING SUMMARY:")
    print(f"   🔄 JSON fields with dots converted to CSV headers with underscores")
    print(f"   📊 {len(FIELDS_TO_CHECK_DOTS)} fields successfully mapped")
    print(f"   📅 Original file date structure preserved")
    print(f"   ✅ Example: battery.voltage -> battery_voltage")
    
    if date_filter:
        print(f"   📁 All CSV files saved to: csv/{date_filter.replace('-', '/')}/")

if __name__ == "__main__":
    main()
