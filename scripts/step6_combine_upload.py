#!/usr/bin/env python3
"""
Step 6: Combine and Upload Regional Data
Combines all processed regional CSV files for a given date and uploads to S3.

Author: SeeSense Data Pipeline
"""

import pandas as pd
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_manager import ConfigManager
from scripts.utils.logger_setup import setup_logger
from scripts.utils.aws_helper import AWSHelper


class RegionalCombiner:
    """Handles combining processed regional data and uploading to S3."""
    
    def __init__(self, config_path=None):
        """Initialize the RegionalCombiner with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logger('regional_combiner', self.config.get_log_config())
        self.aws_helper = AWSHelper(self.config.get_aws_config())
        
        # Set up directories
        self.base_dir = Path(self.config.get('directories.base_dir', str(project_root)))
        self.processed_dir = self.base_dir / self.config.get('directories.processed_dir', 'data/processed')
        self.combined_dir = self.base_dir / self.config.get('directories.combined_dir', 'data/combinedfile')
        
        # AWS settings
        self.bucket = self.config.get('aws.bucket_name')
        self.daily_trips_prefix = self.config.get('aws.daily_trips_prefix')
        
        # Ensure directories exist
        self.combined_dir.mkdir(parents=True, exist_ok=True)
    
    def get_yesterday(self):
        """Get yesterday's date in YYYY-MM-DD format."""
        return (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    def get_today(self):
        """Get today's date in YYYY-MM-DD format for testing."""
        return datetime.utcnow().strftime('%Y-%m-%d')
    
    def prompt_for_date(self, default_date):
        """Prompt user for date to process."""
        try:
            print(f"\nAvailable options:")
            print(f"1. Yesterday: {self.get_yesterday()}")
            print(f"2. Today: {self.get_today()}")
            print(f"3. Custom date")
            
            choice = input(f"Select option (1-3) or press Enter for yesterday [{default_date}]: ").strip()
            
            if choice == '1' or choice == '':
                return self.get_yesterday()
            elif choice == '2':
                return self.get_today()
            elif choice == '3':
                date_input = input("Enter custom date (YYYY-MM-DD): ").strip()
                if date_input:
                    try:
                        datetime.strptime(date_input, '%Y-%m-%d')
                        return date_input
                    except ValueError:
                        print("Invalid date format. Using default.")
                        return default_date
                return default_date
            else:
                return default_date
                
        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user")
            sys.exit(0)
        except Exception as e:
            self.logger.warning(f"Error getting user input: {e}. Using default date: {default_date}")
            return default_date
    
    def find_processable_dates(self) -> List[str]:
        """Find dates that have processed data available for combining."""
        processable_dates = set()
        
        # Check each region for available processed dates
        for region in self.config.get_all_regions():
            processed_region_dir = self.processed_dir / region
            
            if not processed_region_dir.exists():
                continue
            
            # Get all date folders with CSV files
            for date_folder in processed_region_dir.iterdir():
                if not date_folder.is_dir():
                    continue
                
                date_str = date_folder.name
                csv_files = list(date_folder.glob('*.csv'))
                
                if csv_files:
                    processable_dates.add(date_str)
        
        return sorted(list(processable_dates))
    
    def check_s3_destination_exists(self, date_str: str) -> bool:
        """Check if the combined file already exists in S3."""
        try:
            # Convert date format from YYYY-MM-DD to YYYY/MM/DD for S3 path
            date_parts = date_str.split('-')
            year, month, day = date_parts[0], date_parts[1], date_parts[2]
            
            destination_key = f'{self.daily_trips_prefix}year={year}/month={month}/day={day}/{year}{month}{day}_trips.csv'
            
            exists = self.aws_helper.file_exists(destination_key)
            if exists:
                self.logger.info(f"File already exists in S3: s3://{self.bucket}/{destination_key}")
            
            return exists
            
        except Exception as e:
            self.logger.error(f"Error checking S3 destination: {e}")
            return False
    
    def collect_regional_files(self, date_str: str) -> Dict[str, Path]:
        """Collect all processed regional files for a specific date."""
        regional_files = {}
        
        for region in self.config.get_all_regions():
            region_date_dir = self.processed_dir / region / date_str
            
            if not region_date_dir.exists():
                self.logger.debug(f"No processed data directory for {region} on {date_str}")
                continue
            
            # Find CSV files in the region/date directory
            csv_files = list(region_date_dir.glob('*.csv'))
            
            if csv_files:
                # Use the first CSV file found (should be only one)
                regional_files[region] = csv_files[0]
                file_size = csv_files[0].stat().st_size / 1024  # KB
                self.logger.info(f"Found {region} file: {csv_files[0].name} ({file_size:.1f} KB)")
            else:
                self.logger.warning(f"No CSV files found for {region} on {date_str}")
        
        return regional_files
    
    def combine_regional_files(self, regional_files: Dict[str, Path], date_str: str) -> Optional[Path]:
        """Combine regional CSV files into a single file."""
        try:
            if not regional_files:
                self.logger.error("No regional files to combine")
                return None
            
            self.logger.info(f"Combining {len(regional_files)} regional files...")
            
            combined_dataframes = []
            total_rows = 0
            
            # Read and combine all regional files
            for region, file_path in regional_files.items():
                try:
                    self.logger.info(f"Reading {region} file: {file_path}")
                    df = pd.read_csv(file_path)
                    
                    if len(df) == 0:
                        self.logger.warning(f"Empty file: {file_path}")
                        continue
                    
                    # Add region column for tracking
                    df['source_region'] = region
                    combined_dataframes.append(df)
                    total_rows += len(df)
                    
                    self.logger.info(f"  - {region}: {len(df)} rows")
                    
                except Exception as e:
                    self.logger.error(f"Error reading {region} file {file_path}: {e}")
                    continue
            
            if not combined_dataframes:
                self.logger.error("No valid data to combine")
                return None
            
            # Combine all dataframes
            self.logger.info("Concatenating dataframes...")
            combined_df = pd.concat(combined_dataframes, ignore_index=True)
            
            # Sort the combined data
            sort_columns = ['device_id']
            for col in ['timestamp', 'position_timestamp', 'record_seqnum']:
                if col in combined_df.columns:
                    sort_columns.append(col)
            
            combined_df = combined_df.sort_values(by=sort_columns).reset_index(drop=True)
            self.logger.info(f"Combined data sorted by: {sort_columns}")
            
            # Save combined file
            output_filename = f"combined_trips_{date_str.replace('-', '')}.csv"
            output_path = self.combined_dir / output_filename
            
            combined_df.to_csv(output_path, index=False)
            
            # Log summary
            file_size = output_path.stat().st_size / (1024*1024)  # MB
            unique_devices = combined_df['device_id'].nunique() if 'device_id' in combined_df.columns else 0
            trip_breaks = combined_df['trip_break'].sum() if 'trip_break' in combined_df.columns else 0
            
            self.logger.info(f"✅ Combined file created: {output_path}")
            self.logger.info(f"   - Total rows: {len(combined_df)}")
            self.logger.info(f"   - File size: {file_size:.2f} MB")
            self.logger.info(f"   - Unique devices: {unique_devices}")
            self.logger.info(f"   - Trip breaks: {trip_breaks}")
            self.logger.info(f"   - Regions included: {list(regional_files.keys())}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error combining regional files: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def generate_s3_destination_key(self, date_str: str) -> str:
        """Generate the S3 destination key for the combined trips file."""
        # Convert date format from YYYY-MM-DD to YYYY/MM/DD for S3 path
        date_parts = date_str.split('-')
        year, month, day = date_parts[0], date_parts[1], date_parts[2]
        
        return f'{self.daily_trips_prefix}year={year}/month={month}/day={day}/{year}{month}{day}_trips.csv'
    
    def upload_to_s3(self, local_file_path: Path, date_str: str) -> bool:
        """Upload the combined file to S3."""
        try:
            destination_key = self.generate_s3_destination_key(date_str)
            
            self.logger.info(f"Uploading to S3: s3://{self.bucket}/{destination_key}")
            
            # Convert Path object to string for boto3 compatibility
            local_file_str = str(local_file_path)
            
            # Add debug logging
            self.logger.debug(f"Local file path: {local_file_str}")
            self.logger.debug(f"Destination key: {destination_key}")
            self.logger.debug(f"Bucket: {self.bucket}")
            
            success = self.aws_helper.upload_file(local_file_str, destination_key)
            
            if success:
                self.logger.info("✅ File uploaded to S3 successfully")
                self.logger.info(f"   - S3 Location: s3://{self.bucket}/{destination_key}")
                
                # Verify upload by checking file size
                s3_file_size = self.aws_helper.get_file_size(destination_key)
                local_file_size = local_file_path.stat().st_size
                
                if s3_file_size == local_file_size:
                    self.logger.info(f"   - Size verification: ✅ {s3_file_size} bytes")
                else:
                    self.logger.warning(f"   - Size mismatch: local={local_file_size}, s3={s3_file_size}")
                
                return True
            else:
                self.logger.error("❌ Failed to upload file to S3")
                return False
                
        except Exception as e:
            self.logger.error(f"Error uploading to S3: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def cleanup_temporary_files(self, local_file_path: Path):
        """Clean up temporary combined file."""
        try:
            if local_file_path.exists():
                local_file_path.unlink()
                self.logger.info(f"Cleaned up temporary file: {local_file_path}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up temporary file: {e}")
    
    def process_date(self, date_str: str, interactive: bool = True, skip_if_exists: bool = True) -> bool:
        """Process combining and uploading for a specific date."""
        try:
            self.logger.info(f"Starting regional combination for date: {date_str}")
            
            # Check if file already exists in S3
            if skip_if_exists and self.check_s3_destination_exists(date_str):
                if interactive:
                    response = input(f"File already exists in S3 for {date_str}. Overwrite? (y/n) [default: n]: ").strip().lower()
                    if response != 'y':
                        self.logger.info("Skipping due to existing S3 file")
                        return True
                else:
                    self.logger.info("File already exists in S3, skipping (automated mode)")
                    return True
            
            # Collect regional files
            regional_files = self.collect_regional_files(date_str)
            
            if not regional_files:
                self.logger.warning(f"No processed regional files found for date {date_str}")
                self.logger.info("💡 Make sure you've run Step 5 (OSRM Interpolation) first")
                return False
            
            # Combine regional files
            combined_file_path = self.combine_regional_files(regional_files, date_str)
            
            if not combined_file_path:
                self.logger.error("Failed to combine regional files")
                return False
            
            # Upload to S3
            upload_success = self.upload_to_s3(combined_file_path, date_str)
            
            if upload_success:
                # Clean up temporary file in automated mode
                if not interactive:
                    self.cleanup_temporary_files(combined_file_path)
                
                self.logger.info(f"✅ Successfully processed {date_str}")
                return True
            else:
                self.logger.error(f"❌ Failed to upload {date_str}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing date {date_str}: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_interactive(self):
        """Run the combiner in interactive mode."""
        try:
            # Find processable dates
            processable_dates = self.find_processable_dates()
            
            if not processable_dates:
                print("❌ No processable dates found. No processed regional data available.")
                print("💡 Make sure you've run Step 5 (OSRM Interpolation) first.")
                return
            
            print(f"\n📅 Found {len(processable_dates)} dates with processed data:")
            for i, date in enumerate(processable_dates[-5:], 1):  # Show last 5 dates
                print(f"  {i}. {date}")
            
            print("\nOptions:")
            print("1. Process all available dates")
            print("2. Process specific date")
            print("3. Process yesterday's data")
            
            choice = input("Select option (1-3) [default: 3]: ").strip()
            
            if choice == '1':
                # Process all available dates
                for date_str in processable_dates:
                    print(f"\n--- Processing {date_str} ---")
                    self.process_date(date_str, interactive=True)
            elif choice == '2':
                # Process specific date
                print("\nAvailable dates:")
                for i, date in enumerate(processable_dates, 1):
                    print(f"  {i}. {date}")
                
                date_choice = input(f"Select date (1-{len(processable_dates)}): ").strip()
                try:
                    date_idx = int(date_choice) - 1
                    if 0 <= date_idx < len(processable_dates):
                        selected_date = processable_dates[date_idx]
                        self.process_date(selected_date, interactive=True)
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Invalid input")
            else:
                # Process yesterday's data (default)
                yesterday = self.get_yesterday()
                if yesterday in processable_dates:
                    self.process_date(yesterday, interactive=True)
                else:
                    print(f"Yesterday's data ({yesterday}) is not available or already processed")
                    print(f"Available dates: {processable_dates[-3:] if processable_dates else 'None'}")
            
        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Unexpected error in interactive mode: {e}")
            sys.exit(1)
    
    def run_automated(self, date_str: str = None) -> bool:
        """Run the combiner in automated mode (for scheduling)."""
        try:
            if not date_str:
                date_str = self.get_yesterday()
            
            self.logger.info(f"Running automated regional combination for {date_str}")
            success = self.process_date(date_str, interactive=False)
            
            if success:
                self.logger.info("Automated regional combination completed successfully")
                return True
            else:
                self.logger.error("Automated regional combination failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in automated processing: {e}")
            self.logger.debug(traceback.format_exc())
            return False


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Regional Combiner - Step 6 of S2 Data Pipeline')
    parser.add_argument('--date', type=str, help='Date to process (YYYY-MM-DD). Defaults to yesterday.')
    parser.add_argument('--automated', action='store_true', help='Run in automated mode (no user prompts)')
    parser.add_argument('--force', action='store_true', help='Force overwrite if file exists in S3')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        combiner = RegionalCombiner(args.config)
        
        if args.automated:
            success = combiner.run_automated(args.date)
            sys.exit(0 if success else 1)
        else:
            if args.date:
                skip_if_exists = not args.force
                combiner.process_date(args.date, interactive=True, skip_if_exists=skip_if_exists)
            else:
                combiner.run_interactive()
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
