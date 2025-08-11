#!/usr/bin/env python3
"""
Step 3: Daily CSV Combiner
Downloads individual CSV files from S3, combines them into a single daily CSV file,
and uploads the result back to S3.

Author: SeeSense Data Pipeline
"""

import boto3
import os
import json
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import concurrent.futures
import sys
import traceback
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_manager import ConfigManager
from scripts.utils.logger_setup import setup_logger
from scripts.utils.aws_helper import AWSHelper


class DailyCombiner:
    """Handles downloading, combining, and uploading daily CSV files."""
    
    def __init__(self, config_path=None):
        """Initialize the DailyCombiner with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logger('daily_combiner', self.config.get_log_config())
        self.aws_helper = AWSHelper(self.config.get_aws_config())
        
        # Set up directories
        self.base_dir = Path(self.config.get('directories.base_dir', str(project_root)))
        self.download_dir = self.base_dir / self.config.get('directories.download_dir', 'data/downloadedfiles')
        self.combined_dir = self.base_dir / self.config.get('directories.combined_dir', 'data/combinedfile')
        
        # Processing settings
        self.max_workers = self.config.get('processing.max_workers', 10)
        self.bucket = self.config.get('aws.bucket_name')
        
    def get_yesterday(self):
        """Get yesterday's date in YYYY/MM/DD format."""
        return (datetime.utcnow() - timedelta(days=1)).strftime('%Y/%m/%d')
    
    def prompt_for_date(self, default_date):
        """Prompt user for date to process, with default being yesterday."""
        try:
            date_input = input(f"Enter the date to process (YYYY/MM/DD) [default: {default_date}]: ")
            return date_input.strip() or default_date
        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user")
            sys.exit(0)
        except Exception as e:
            self.logger.warning(f"Error getting user input: {e}. Using default date: {default_date}")
            return default_date
    
    def clean_directory(self, directory):
        """Clean up directory contents with user confirmation."""
        directory = Path(directory)
        
        if directory.exists() and any(directory.iterdir()):
            try:
                prompt = input(f"The directory {directory} is not empty. Do you want to clean it up? (y/n) [default: y]: ")
                if prompt.lower() in ['n', 'no']:
                    self.logger.info(f"Skipping cleanup of {directory}")
                    return False
                    
                # Clean up directory
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        
                self.logger.info(f"Cleaned directory: {directory}")
                        
            except KeyboardInterrupt:
                self.logger.info("Process interrupted by user")
                sys.exit(0)
            except Exception as e:
                self.logger.error(f"Error cleaning directory {directory}: {e}")
                return False
        
        # Ensure directory exists
        directory.mkdir(parents=True, exist_ok=True)
        return True
    
    def download_file(self, key):
        """Download a single file from S3."""
        try:
            file_name = key.split('/')[-1]
            local_path = self.download_dir / file_name
            
            self.aws_helper.download_file(key, str(local_path))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {key}: {e}")
            return False
    
    def download_s3_files(self, keys):
        """Download multiple files from S3 using ThreadPoolExecutor."""
        self.logger.info(f"Starting download of {len(keys)} files...")
        
        successful_downloads = 0
        failed_downloads = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_key = {executor.submit(self.download_file, key): key for key in keys}
            
            # Process completed downloads with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_key), 
                             total=len(future_to_key), 
                             desc="Downloading files"):
                key = future_to_key[future]
                try:
                    if future.result():
                        successful_downloads += 1
                    else:
                        failed_downloads.append(key)
                except Exception as e:
                    self.logger.error(f"Download thread error for {key}: {e}")
                    failed_downloads.append(key)
        
        self.logger.info(f"Downloads completed: {successful_downloads} successful, {len(failed_downloads)} failed")
        
        if failed_downloads:
            self.logger.warning(f"Failed downloads: {failed_downloads[:5]}...")  # Log first 5 failures
            
        return successful_downloads > 0
    
    def combine_csv_files(self, output_file):
        """Combine all CSV files in download directory into a single file."""
        output_file = Path(output_file)
        csv_files = list(self.download_dir.glob('*.csv'))
        
        if not csv_files:
            self.logger.error("No CSV files found to combine")
            return False
        
        self.logger.info(f"Combining {len(csv_files)} CSV files...")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                first_file = True
                
                for csv_file in tqdm(csv_files, desc="Combining files"):
                    try:
                        with open(csv_file, 'r', encoding='utf-8') as infile:
                            if first_file:
                                # Write the entire first file including header
                                outfile.write(infile.read())
                                first_file = False
                            else:
                                # Skip header row for subsequent files
                                infile.readline()
                                outfile.write(infile.read())
                                
                    except Exception as e:
                        self.logger.error(f"Error processing file {csv_file}: {e}")
                        continue
            
            # Verify the combined file was created and has content
            if output_file.exists() and output_file.stat().st_size > 0:
                self.logger.info(f"Successfully combined CSV files into: {output_file}")
                self.logger.info(f"Combined file size: {output_file.stat().st_size / (1024*1024):.2f} MB")
                return True
            else:
                self.logger.error("Combined file is empty or was not created")
                return False
                
        except Exception as e:
            self.logger.error(f"Error combining CSV files: {e}")
            return False
    
    def generate_destination_key(self, date_str):
        """Generate the S3 destination key for the combined file."""
        date_parts = date_str.split('/')
        year, month, day = date_parts[0], date_parts[1], date_parts[2]
        
        daily_csv_prefix = self.config.get('aws.daily_csv_prefix')
        return f'{daily_csv_prefix}year={year}/month={month}/day={day}/{year}{month}{day}.csv'
    
    def process_date(self, date_str, interactive=True):
        """Process CSV files for a specific date."""
        self.logger.info(f"Starting processing for date: {date_str}")
        
        try:
            # Set up directories
            if interactive and not self.clean_directory(self.download_dir):
                return False
            
            self.combined_dir.mkdir(parents=True, exist_ok=True)
            
            # List files to download
            source_prefix = f"{self.config.get('aws.source_prefix')}{date_str}/"
            self.logger.info(f"Listing files in s3://{self.bucket}/{source_prefix}...")
            
            files_to_process = self.aws_helper.list_files(source_prefix)
            
            if not files_to_process:
                self.logger.warning(f"No files found for date {date_str}")
                return False
                
            self.logger.info(f"Found {len(files_to_process)} files to process")
            
            # Download files
            if not self.download_s3_files(files_to_process):
                self.logger.error("Failed to download files")
                return False
            
            # Combine CSV files
            combined_csv_path = self.combined_dir / 'combined.csv'
            if not self.combine_csv_files(combined_csv_path):
                self.logger.error("Failed to combine CSV files")
                return False
            
            # Upload combined file
            destination_key = self.generate_destination_key(date_str)
            self.logger.info(f"Uploading combined file to s3://{self.bucket}/{destination_key}...")
            
            if self.aws_helper.upload_file(str(combined_csv_path), destination_key):
                self.logger.info("Successfully uploaded combined file to S3")
                
                # Clean up temporary files if in non-interactive mode
                if not interactive:
                    self._cleanup_temp_files()
                    
                return True
            else:
                self.logger.error("Failed to upload combined file to S3")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing date {date_str}: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def _cleanup_temp_files(self):
        """Clean up temporary downloaded files."""
        try:
            for file_path in self.download_dir.glob('*.csv'):
                file_path.unlink()
            self.logger.info("Cleaned up temporary download files")
        except Exception as e:
            self.logger.warning(f"Error cleaning up temp files: {e}")
    
    def run_interactive(self):
        """Run the combiner in interactive mode."""
        try:
            default_date = self.get_yesterday()
            date_to_run = self.prompt_for_date(default_date)
            
            success = self.process_date(date_to_run, interactive=True)
            
            if success:
                print("✅ Process completed successfully!")
            else:
                print("❌ Process failed. Check logs for details.")
                sys.exit(1)
                
        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Unexpected error in interactive mode: {e}")
            sys.exit(1)
    
    def run_automated(self, date_str=None):
        """Run the combiner in automated mode (for scheduling)."""
        try:
            if not date_str:
                date_str = self.get_yesterday()
            
            self.logger.info(f"Running automated processing for {date_str}")
            success = self.process_date(date_str, interactive=False)
            
            if success:
                self.logger.info("Automated processing completed successfully")
                return True
            else:
                self.logger.error("Automated processing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in automated processing: {e}")
            self.logger.debug(traceback.format_exc())
            return False


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily CSV Combiner - Step 3 of S2 Data Pipeline')
    parser.add_argument('--date', type=str, help='Date to process (YYYY/MM/DD). Defaults to yesterday.')
    parser.add_argument('--automated', action='store_true', help='Run in automated mode (no user prompts)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        combiner = DailyCombiner(args.config)
        
        if args.automated:
            success = combiner.run_automated(args.date)
            sys.exit(0 if success else 1)
        else:
            combiner.run_interactive()
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
