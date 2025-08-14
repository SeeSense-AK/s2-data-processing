#!/usr/bin/env python3
"""
Step 4: Device Bifurcation
Splits daily combined CSV files by device regions based on device name prefixes.

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


class DeviceBifurcator:
    """Handles splitting CSV files by device regions."""
    
    def __init__(self, config_path=None):
        """Initialize the DeviceBifurcator with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logger('device_bifurcator', self.config.get_log_config())
        self.aws_helper = AWSHelper(self.config.get_aws_config())
        
        # Set up directories
        self.base_dir = Path(self.config.get('directories.base_dir', str(project_root)))
        self.combined_dir = self.base_dir / self.config.get('directories.combined_dir', 'data/combinedfile')
        self.preprocessed_dir = self.base_dir / self.config.get('directories.preprocessed_dir', 'data/preprocessed')
        
        # Settings
        self.bucket = self.config.get('aws.bucket_name')
        self.retention_days = self.config.get('processing.retention_days', 2)
        
        # Ensure directories exist
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create region directories
        for region in self.config.get_all_regions():
            region_dir = self.preprocessed_dir / region
            region_dir.mkdir(parents=True, exist_ok=True)
    
    def get_yesterday(self):
        """Get yesterday's date in YYYY/MM/DD format."""
        return (datetime.utcnow() - timedelta(days=1)).strftime('%Y/%m/%d')
    
    def get_today(self):
        """Get today's date in YYYY/MM/DD format for testing."""
        return datetime.utcnow().strftime('%Y/%m/%d')
    
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
                date_input = input("Enter custom date (YYYY/MM/DD): ").strip()
                if date_input:
                    try:
                        datetime.strptime(date_input, '%Y/%m/%d')
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
    
    def find_local_combined_csv(self, date_str: str) -> Optional[Path]:
        """Find the local combined CSV file for the specified date."""
        try:
            # Convert date format from YYYY/MM/DD to YYYYMMDD
            date_compact = date_str.replace('-', '')
            expected_filename = f"combined_{date_compact}.csv"
            expected_path = self.combined_dir / expected_filename
            
            self.logger.info(f"Looking for local combined CSV: {expected_path}")
            
            if expected_path.exists():
                file_size = expected_path.stat().st_size / (1024*1024)  # MB
                self.logger.info(f"✅ Found local combined CSV: {expected_path} ({file_size:.2f} MB)")
                return expected_path
            else:
                self.logger.warning(f"❌ Local combined CSV not found: {expected_path}")
                
                # List available files in combined directory
                available_files = list(self.combined_dir.glob("combined_*.csv"))
                if available_files:
                    self.logger.info("📁 Available combined CSV files:")
                    for file in available_files:
                        # Extract date from filename
                        filename = file.stem  # Remove .csv extension
                        if filename.startswith('combined_') and len(filename) == 17:  # combined_YYYYMMDD
                            date_part = filename[9:]  # Extract YYYYMMDD
                            formatted_date = f"{date_part[:4]}/{date_part[4:6]}/{date_part[6:8]}"
                            file_size = file.stat().st_size / (1024*1024)
                            self.logger.info(f"   📄 {file.name} -> Date: {formatted_date} ({file_size:.2f} MB)")
                else:
                    self.logger.warning("📁 No combined CSV files found in combined directory")
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error finding local combined CSV: {e}")
            return None
    
    def verify_file_date_match(self, csv_path: Path, expected_date: str) -> bool:
        """Verify that the CSV file corresponds to the expected date."""
        try:
            # Extract date from filename
            filename = csv_path.stem  # Remove .csv extension
            if filename.startswith('combined_') and len(filename) == 17:  # combined_YYYYMMDD
                file_date_compact = filename[9:]  # Extract YYYYMMDD
                expected_date_compact = expected_date.replace('/', '')
                
                if file_date_compact == expected_date_compact:
                    self.logger.info(f"✅ File date matches expected date: {expected_date}")
                    return True
                else:
                    file_date_formatted = f"{file_date_compact[:4]}/{file_date_compact[4:6]}/{file_date_compact[6:8]}"
                    self.logger.warning(f"❌ File date mismatch: file={file_date_formatted}, expected={expected_date}")
                    return False
            else:
                self.logger.warning(f"❌ Invalid filename format: {filename}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error verifying file date: {e}")
            return False
    
    def get_device_region(self, device_name: str) -> Optional[str]:
        """Get the region for a device based on its name prefix."""
        if pd.isna(device_name) or not device_name:
            return None
        
        region = self.config.get_region_for_device(str(device_name).strip())
        if not region:
            self.logger.warning(f"No region found for device: {device_name}")
        
        return region
    
    def analyze_csv_structure(self, csv_path: Path) -> Dict:
        """Analyze the CSV file structure to understand the data."""
        try:
            # Read a small sample to understand structure
            sample_df = pd.read_csv(csv_path, nrows=100)
            
            info = {
                'total_rows_sample': len(sample_df),
                'columns': list(sample_df.columns),
                'has_device_name': 'device_name' in sample_df.columns,
                'device_name_samples': []
            }
            
            if info['has_device_name']:
                # Get unique device name samples
                unique_devices = sample_df['device_name'].dropna().unique()
                info['device_name_samples'] = list(unique_devices[:10])  # First 10 samples
                
                # Test region mapping
                region_counts = {}
                for device in unique_devices:
                    region = self.get_device_region(device)
                    if region:
                        region_counts[region] = region_counts.get(region, 0) + 1
                    else:
                        region_counts['unknown'] = region_counts.get('unknown', 0) + 1
                
                info['region_distribution'] = region_counts
            
            self.logger.info(f"CSV Analysis: {info}")
            return info
            
        except Exception as e:
            self.logger.error(f"Error analyzing CSV structure: {e}")
            return {'error': str(e)}
    
    def bifurcate_csv_by_region(self, csv_path: Path, date_str: str) -> Dict[str, int]:
        """Split CSV file by device regions."""
        try:
            self.logger.info(f"Starting bifurcation of {csv_path}")
            
            # Analyze CSV structure first
            analysis = self.analyze_csv_structure(csv_path)
            if 'error' in analysis:
                raise ValueError(f"CSV analysis failed: {analysis['error']}")
            
            if not analysis.get('has_device_name'):
                raise ValueError("CSV file does not contain 'device_name' column")
            
            # Read the full CSV file
            self.logger.info("Reading CSV file...")
            df = pd.read_csv(csv_path)
            
            total_rows = len(df)
            self.logger.info(f"Total rows to process: {total_rows}")
            
            if total_rows == 0:
                self.logger.warning("CSV file is empty")
                return {}
            
            # Create date subdirectories for each region
            date_folder = date_str.replace('/', '-')  # Convert 2025/08/11 to 2025-08-11
            
            # Process data by regions
            region_counts = {}
            unmatched_devices = set()
            
            # Group by device region
            self.logger.info("Grouping devices by region...")
            
            for index, row in df.iterrows():
                device_name = row.get('device_name')
                region = self.get_device_region(device_name)
                
                if region:
                    if region not in region_counts:
                        region_counts[region] = []
                    region_counts[region].append(index)
                else:
                    unmatched_devices.add(str(device_name))
            
            # Log unmatched devices
            if unmatched_devices:
                self.logger.warning(f"Unmatched devices: {list(unmatched_devices)[:10]}...")  # Show first 10
            
            # Write region-specific CSV files
            file_counts = {}
            
            for region, row_indices in region_counts.items():
                region_df = df.iloc[row_indices]
                
                # Create region/date directory
                region_dir = self.preprocessed_dir / region / date_folder
                region_dir.mkdir(parents=True, exist_ok=True)
                
                # Save CSV file
                output_file = region_dir / f"{region}_{date_str.replace('/', '')}.csv"
                region_df.to_csv(output_file, index=False)
                
                file_counts[region] = len(region_df)
                self.logger.info(f"Created {output_file} with {len(region_df)} rows")
            
            # Log summary
            total_processed = sum(file_counts.values())
            self.logger.info(f"Bifurcation complete:")
            self.logger.info(f"  Total rows processed: {total_processed}/{total_rows}")
            self.logger.info(f"  Regions created: {list(file_counts.keys())}")
            self.logger.info(f"  Unmatched devices: {len(unmatched_devices)}")
            
            return file_counts
            
        except Exception as e:
            self.logger.error(f"Error bifurcating CSV: {e}")
            self.logger.debug(traceback.format_exc())
            return {}
    
    def cleanup_old_data(self):
        """Remove preprocessed data older than retention_days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            removed_count = 0
            
            for region_dir in self.preprocessed_dir.iterdir():
                if not region_dir.is_dir():
                    continue
                
                for date_dir in region_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                    
                    try:
                        # Parse date from directory name (format: 2025-08-11)
                        date_obj = datetime.strptime(date_dir.name, '%Y-%m-%d')
                        
                        if date_obj < cutoff_date:
                            self.logger.info(f"Removing old data: {date_dir}")
                            shutil.rmtree(date_dir)
                            removed_count += 1
                            
                    except ValueError:
                        # Skip directories that don't match date format
                        continue
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old date directories")
            else:
                self.logger.info("No old data to clean up")
                
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def process_date(self, date_str: str, interactive: bool = True) -> bool:
        """Process bifurcation for a specific date."""
        try:
            self.logger.info(f"Starting device bifurcation for date: {date_str}")
            
            # Find local combined CSV file
            csv_path = self.find_local_combined_csv(date_str)
            if not csv_path:
                self.logger.error(f"No local combined CSV file found for date {date_str}")
                self.logger.info("💡 Tip: Make sure you've run Step 3 first to create the combined CSV file")
                return False
            
            # Verify the file corresponds to the expected date
            if not self.verify_file_date_match(csv_path, date_str):
                if interactive:
                    response = input(f"Date mismatch detected. Continue anyway? (y/n) [default: n]: ").strip().lower()
                    if response != 'y':
                        self.logger.info("Process cancelled by user due to date mismatch")
                        return False
                else:
                    self.logger.error("Date mismatch detected in automated mode. Stopping.")
                    return False
            
            # Bifurcate by regions
            file_counts = self.bifurcate_csv_by_region(csv_path, date_str)
            
            if not file_counts:
                self.logger.error("Bifurcation failed - no regional files created")
                return False
            
            # Clean up old data
            self.cleanup_old_data()
            
            self.logger.info("✅ Device bifurcation completed successfully")
            self.logger.info(f"📁 Regional files created in {self.preprocessed_dir}/[region]/{date_str.replace('/', '-')}/")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing date {date_str}: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_interactive(self):
        """Run the bifurcator in interactive mode."""
        try:
            default_date = self.get_yesterday()
            date_to_run = self.prompt_for_date(default_date)
            
            success = self.process_date(date_to_run, interactive=True)
            
            if success:
                print("✅ Device bifurcation completed successfully!")
                
                # Show results
                date_folder = date_to_run.replace('/', '-')
                print(f"\n📁 Regional files created in data/preprocessed/[region]/{date_folder}/")
                
                # List created files
                for region in self.config.get_all_regions():
                    region_dir = self.preprocessed_dir / region / date_folder
                    if region_dir.exists():
                        files = list(region_dir.glob('*.csv'))
                        if files:
                            file_size = files[0].stat().st_size / 1024  # KB
                            print(f"  📄 {region}: {files[0].name} ({file_size:.1f} KB)")
            else:
                print("❌ Device bifurcation failed. Check logs for details.")
                sys.exit(1)
                
        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Unexpected error in interactive mode: {e}")
            sys.exit(1)
    
    def run_automated(self, date_str: str = None) -> bool:
        """Run the bifurcator in automated mode (for scheduling)."""
        try:
            if not date_str:
                date_str = self.get_yesterday()
            
            self.logger.info(f"Running automated device bifurcation for {date_str}")
            success = self.process_date(date_str, interactive=False)
            
            if success:
                self.logger.info("Automated device bifurcation completed successfully")
                return True
            else:
                self.logger.error("Automated device bifurcation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in automated processing: {e}")
            self.logger.debug(traceback.format_exc())
            return False


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Device Bifurcation - Step 5 of S2 Data Pipeline')
    parser.add_argument('--date', type=str, help='Date to process (YYYY/MM/DD). Defaults to yesterday.')
    parser.add_argument('--automated', action='store_true', help='Run in automated mode (no user prompts)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        bifurcator = DeviceBifurcator(args.config)
        
        if args.automated:
            success = bifurcator.run_automated(args.date)
            sys.exit(0 if success else 1)
        else:
            bifurcator.run_interactive()
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
