#!/usr/bin/env python3
"""
Step 7: Abnormal Events Detection
Detects abnormal driving events (braking, swerving, potholes) using accelerometer data.
Uses hybrid quantile + axis dominance + MAD method for robust event detection.

Author: SeeSense Data Pipeline
"""

import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import median_abs_deviation

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_manager import ConfigManager
from scripts.utils.logger_setup import setup_logger
from scripts.utils.aws_helper import AWSHelper
from scripts.utils.date_utils import normalize_date_format, get_yesterday_formats, get_today_formats


class AbnormalEventsDetector:
    """Handles detection of abnormal driving events from accelerometer data."""
    
    def __init__(self, config_path=None):
        """Initialize the AbnormalEventsDetector with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logger('abnormal_events', self.config.get_log_config())
        self.aws_helper = AWSHelper(self.config.get_aws_config())
        
        # Set up directories
        self.base_dir = Path(self.config.get('directories.base_dir', str(project_root)))
        self.final_output_dir = self.base_dir / self.config.get('directories.final_output_dir', 'data/finaloutput')
        self.abnormal_events_dir = self.base_dir / 'data/abnormal-events'
        
        # AWS settings
        self.bucket = self.config.get('aws.bucket_name')
        self.abnormal_events_prefix = 'test-abnormal-events-csv/'
        
        # Detection parameters (configurable)
        self.quantile_threshold = self.config.get('abnormal_events.quantile_threshold', 95)
        self.mad_threshold = self.config.get('abnormal_events.mad_threshold', 3)
        self.axis_dominance_factor = self.config.get('abnormal_events.axis_dominance_factor', 2)
        
        # Required accelerometer columns
        self.required_accel_columns = ['ain.12', 'ain.13', 'ain.14', 'ain.15', 'ain.16', 'ain.17']
        self.peak_columns = ['ain.12', 'ain.13', 'ain.14']  # peak_x, peak_y, peak_z
        
        # Ensure directories exist
        self.abnormal_events_dir.mkdir(parents=True, exist_ok=True)
    
    def check_accelerometer_columns(self, df: pd.DataFrame) -> bool:
        """Check if the DataFrame has required accelerometer columns."""
        missing_columns = [col for col in self.required_accel_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.info(f"❌ Missing accelerometer columns: {missing_columns}")
            self.logger.info("No accelerometer readings found - skipping abnormal events detection")
            return False
        
        self.logger.info("✅ All required accelerometer columns found")
        return True
    
    def prepare_accelerometer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean accelerometer data for analysis."""
        # Rename columns for easier processing
        df_clean = df.rename(columns={
            'ain.12': 'peak_x',
            'ain.13': 'peak_y', 
            'ain.14': 'peak_z',
            'ain.15': 'avg_x',
            'ain.16': 'avg_y',
            'ain.17': 'avg_z'
        }).copy()
        
        # Remove rows with missing coordinates
        df_clean = df_clean.dropna(subset=['snapped_lat', 'snapped_lon'])
        
        # Remove (0,0) coordinates
        df_clean = df_clean.query('snapped_lat != 0 or snapped_lon != 0')
        
        # Remove non-event rows (where all peak values are 0 or NaN)
        df_active = df_clean[
            (df_clean[['peak_x', 'peak_y', 'peak_z']] > 0).any(axis=1)
        ].copy().reset_index(drop=True)
        
        self.logger.info(f"Prepared accelerometer data: {len(df_active)} active events from {len(df)} total rows")
        return df_active
    
    def is_dominant_axis(self, row: pd.Series, axis: str) -> bool:
        """Check if the specified axis is dominant (2x greater than others)."""
        if axis == 'x':
            return (row['peak_x'] > self.axis_dominance_factor * row['peak_y'] and 
                   row['peak_x'] > self.axis_dominance_factor * row['peak_z'])
        elif axis == 'y':
            return (row['peak_y'] > self.axis_dominance_factor * row['peak_x'] and 
                   row['peak_y'] > self.axis_dominance_factor * row['peak_z'])
        elif axis == 'z':
            return (row['peak_z'] > self.axis_dominance_factor * row['peak_x'] and 
                   row['peak_z'] > self.axis_dominance_factor * row['peak_y'])
        return False
    
    def is_mad_outlier(self, series: pd.Series, value: float) -> bool:
        """Check if a value is a MAD (Median Absolute Deviation) outlier."""
        mad = median_abs_deviation(series)
        median = np.median(series)
        
        # Avoid division by zero
        if mad == 0:
            return False
            
        return abs(value - median) / mad > self.mad_threshold
    
    def calculate_severity_score(self, peak_value: float, threshold: float, max_value: float) -> int:
        """Calculate severity score (1-10) based on peak value relative to threshold."""
        if peak_value <= threshold:
            return 1
        
        # Scale from threshold to max observed value
        severity_ratio = (peak_value - threshold) / (max_value - threshold) if max_value > threshold else 0
        
        # Map to 1-10 scale, with 10 being the most severe
        severity = int(2 + (severity_ratio * 8))
        return min(max(severity, 1), 10)
    
    def detect_abnormal_events(self, df_active: pd.DataFrame) -> List[Dict]:
        """
        Detect abnormal events using hybrid quantile + axis dominance + MAD method.
        
        Returns:
            List of event dictionaries with event details and severity scores
        """
        if len(df_active) < 10:
            self.logger.warning("Insufficient data for reliable abnormal event detection")
            return []
        
        # Calculate thresholds using quantile method
        thresholds = {
            'x': np.percentile(df_active['peak_x'], self.quantile_threshold),
            'y': np.percentile(df_active['peak_y'], self.quantile_threshold),
            'z': np.percentile(df_active['peak_z'], self.quantile_threshold)
        }
        
        # Calculate max values for severity scoring
        max_values = {
            'x': df_active['peak_x'].max(),
            'y': df_active['peak_y'].max(),
            'z': df_active['peak_z'].max()
        }
        
        self.logger.info(f"Detection thresholds - X: {thresholds['x']:.2f}, Y: {thresholds['y']:.2f}, Z: {thresholds['z']:.2f}")
        
        events = []
        
        for idx, row in df_active.iterrows():
            # Check X-axis (hard braking)
            if (row['peak_x'] > thresholds['x'] and 
                self.is_dominant_axis(row, 'x') and
                self.is_mad_outlier(df_active['peak_x'], row['peak_x'])):
                
                severity = self.calculate_severity_score(row['peak_x'], thresholds['x'], max_values['x'])
                events.append({
                    'event_type': 'hard_brake',
                    'original_index': idx,
                    'latitude': row['snapped_lat'],
                    'longitude': row['snapped_lon'],
                    'peak_value': row['peak_x'],
                    'severity': severity,
                    'timestamp': row.get('timestamp', ''),
                    'device_id': row.get('device_id', ''),
                    'trip_id': row.get('trip_id', '')
                })
            
            # Check Y-axis (swerving)
            if (row['peak_y'] > thresholds['y'] and 
                self.is_dominant_axis(row, 'y') and
                self.is_mad_outlier(df_active['peak_y'], row['peak_y'])):
                
                severity = self.calculate_severity_score(row['peak_y'], thresholds['y'], max_values['y'])
                events.append({
                    'event_type': 'swerve',
                    'original_index': idx,
                    'latitude': row['snapped_lat'],
                    'longitude': row['snapped_lon'],
                    'peak_value': row['peak_y'],
                    'severity': severity,
                    'timestamp': row.get('timestamp', ''),
                    'device_id': row.get('device_id', ''),
                    'trip_id': row.get('trip_id', '')
                })
            
            # Check Z-axis (potholes)
            if (row['peak_z'] > thresholds['z'] and 
                self.is_dominant_axis(row, 'z') and
                self.is_mad_outlier(df_active['peak_z'], row['peak_z'])):
                
                severity = self.calculate_severity_score(row['peak_z'], thresholds['z'], max_values['z'])
                events.append({
                    'event_type': 'pothole',
                    'original_index': idx,
                    'latitude': row['snapped_lat'],
                    'longitude': row['snapped_lon'],
                    'peak_value': row['peak_z'],
                    'severity': severity,
                    'timestamp': row.get('timestamp', ''),
                    'device_id': row.get('device_id', ''),
                    'trip_id': row.get('trip_id', '')
                })
        
        # Log summary by event type and severity
        event_summary = {}
        severity_distribution = {}
        
        for event in events:
            event_type = event['event_type']
            severity = event['severity']
            
            event_summary[event_type] = event_summary.get(event_type, 0) + 1
            severity_key = f"severity_{severity}"
            severity_distribution[severity_key] = severity_distribution.get(severity_key, 0) + 1
        
        self.logger.info(f"Detected {len(events)} abnormal events:")
        for event_type, count in event_summary.items():
            self.logger.info(f"  - {event_type}: {count}")
        
        self.logger.info("Severity distribution:")
        for severity_key in sorted(severity_distribution.keys()):
            self.logger.info(f"  - {severity_key}: {severity_distribution[severity_key]}")
        
        return events
    
    def create_events_dataframe(self, events: List[Dict]) -> pd.DataFrame:
        """Convert events list to DataFrame for saving."""
        if not events:
            return pd.DataFrame()
        
        events_df = pd.DataFrame(events)
        
        # Reorder columns for better readability
        column_order = [
            'event_type', 'severity', 'timestamp', 'device_id', 'trip_id',
            'latitude', 'longitude', 'peak_value', 'original_index'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in events_df.columns]
        events_df = events_df[available_columns]
        
        return events_df
    
    def find_finaloutput_file(self, date_str: str) -> Optional[Path]:
        """Find the finaloutput file for the specified date."""
        # Use utility function to normalize date format
        _, date_folder, date_compact = normalize_date_format(date_str)
        
        # Look for file with expected naming pattern
        expected_file = self.final_output_dir / f"{date_compact}_trips.csv"
        
        if expected_file.exists():
            return expected_file
        
        # Fallback: look for any CSV files for that date
        csv_files = list(self.final_output_dir.glob(f"*{date_compact}*.csv"))
        if csv_files:
            self.logger.warning(f"Using fallback finaloutput file: {csv_files[0].name}")
            return csv_files[0]
        
        return None
    
    def generate_s3_destination_key(self, date_str: str) -> str:
        """Generate the S3 destination key for the abnormal events file."""
        # Use utility function to normalize date format
        _, _, date_compact = normalize_date_format(date_str)
        
        # Extract components for S3 path  
        year = date_compact[:4]
        month = date_compact[4:6]
        day = date_compact[6:8]
        
        return f'{self.abnormal_events_prefix}year={year}/month={month}/day={day}/{date_compact}_abnormal_events.csv'
    
    def upload_to_s3(self, local_file_path: Path, date_str: str) -> bool:
        """Upload the abnormal events file to S3."""
        try:
            destination_key = self.generate_s3_destination_key(date_str)
            
            self.logger.info(f"Uploading to S3: s3://{self.bucket}/{destination_key}")
            
            success = self.aws_helper.upload_file(str(local_file_path), destination_key)
            
            if success:
                self.logger.info("✅ Abnormal events file uploaded to S3 successfully")
                
                # Verify upload by checking file size
                s3_file_size = self.aws_helper.get_file_size(destination_key)
                local_file_size = local_file_path.stat().st_size
                
                if s3_file_size == local_file_size:
                    self.logger.info(f"   - Size verification: ✅ {s3_file_size} bytes")
                else:
                    self.logger.warning(f"   - Size mismatch: local={local_file_size}, s3={s3_file_size}")
                
                return True
            else:
                self.logger.error("❌ Failed to upload abnormal events file to S3")
                return False
                
        except Exception as e:
            self.logger.error(f"Error uploading abnormal events to S3: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def process_date(self, date_str: str, interactive: bool = True) -> bool:
        """Process abnormal events detection for a specific date."""
        try:
            # Use utility function to normalize date format
            _, date_folder, date_compact = normalize_date_format(date_str)
            
            self.logger.info(f"🔍 Starting abnormal events detection for {date_str}")
            
            # Find the finaloutput file for this date
            input_file = self.find_finaloutput_file(date_str)
            
            if not input_file:
                self.logger.warning(f"❌ No finaloutput file found for {date_str}")
                self.logger.info("Skipping abnormal events detection - this will not break the pipeline")
                return True  # Return True to not break the pipeline
            
            self.logger.info(f"📁 Processing file: {input_file}")
            
            # Load the CSV file
            df = pd.read_csv(input_file)
            self.logger.info(f"Loaded {len(df)} rows from finaloutput file")
            
            # Check if accelerometer columns are present
            if not self.check_accelerometer_columns(df):
                return True  # Return True to not break the pipeline
            
            # Prepare accelerometer data
            df_active = self.prepare_accelerometer_data(df)
            
            if len(df_active) == 0:
                self.logger.warning("No active accelerometer data found")
                return True  # Return True to not break the pipeline
            
            # Detect abnormal events
            events = self.detect_abnormal_events(df_active)
            
            if not events:
                self.logger.info("No abnormal events detected")
                # Still create an empty file for consistency
                events_df = pd.DataFrame()
            else:
                events_df = self.create_events_dataframe(events)
            
            # Save locally
            local_output_file = self.abnormal_events_dir / f"{date_compact}_abnormal_events.csv"
            
            if len(events_df) > 0:
                events_df.to_csv(local_output_file, index=False)
                self.logger.info(f"✅ Abnormal events saved locally: {local_output_file}")
                self.logger.info(f"   - Events detected: {len(events_df)}")
            else:
                # Create empty file
                pd.DataFrame().to_csv(local_output_file, index=False)
                self.logger.info(f"✅ Empty abnormal events file created: {local_output_file}")
            
            # Upload to S3
            upload_success = self.upload_to_s3(local_output_file, date_str)
            
            if upload_success:
                self.logger.info("✅ Step 7: Abnormal Events Detection completed successfully")
            else:
                self.logger.warning("⚠️ Local processing completed but S3 upload failed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in abnormal events detection: {e}")
            self.logger.debug(traceback.format_exc())
            self.logger.info("Returning True to not break the pipeline")
            return True  # Return True to not break the pipeline
    
    def run_automated(self, date_str: str) -> bool:
        """Run abnormal events detection in automated mode."""
        return self.process_date(date_str, interactive=False)
    
    def run_interactive(self):
        """Run abnormal events detection in interactive mode."""
        try:
            print("🚀 Abnormal Events Detection (Step 7)")
            print("=" * 50)
            
            # Get date options
            yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
            today = datetime.utcnow().strftime('%Y-%m-%d')
            
            print(f"\nDate options:")
            print(f"1. Yesterday: {yesterday}")
            print(f"2. Today: {today}")
            print(f"3. Custom date")
            
            choice = input(f"Select option (1-3) or press Enter for yesterday [{yesterday}]: ").strip()
            
            if choice == '1' or choice == '':
                date_str = yesterday
            elif choice == '2':
                date_str = today
            elif choice == '3':
                date_input = input("Enter custom date (YYYY-MM-DD): ").strip()
                if date_input:
                    try:
                        datetime.strptime(date_input, '%Y-%m-%d')
                        date_str = date_input
                    except ValueError:
                        print("Invalid date format. Using yesterday.")
                        date_str = yesterday
                else:
                    date_str = yesterday
            else:
                date_str = yesterday
            
            success = self.process_date(date_str, interactive=True)
            
            if success:
                print("✅ Abnormal events detection completed successfully!")
            else:
                print("❌ Abnormal events detection failed!")
            
            return success
            
        except KeyboardInterrupt:
            self.logger.info("❌ Process interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in interactive mode: {e}")
            self.logger.debug(traceback.format_exc())
            return False


def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 7: Abnormal Events Detection')
    parser.add_argument('--date', type=str, help='Date to process (YYYY-MM-DD format)')
    parser.add_argument('--automated', action='store_true', help='Run in automated mode')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = AbnormalEventsDetector(args.config)
    
    if args.automated:
        # Automated mode - use yesterday if no date specified
        date_str = args.date or (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
        success = detector.run_automated(date_str)
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        if args.date:
            success = detector.process_date(args.date, interactive=True)
            sys.exit(0 if success else 1)
        else:
            success = detector.run_interactive()
            sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
