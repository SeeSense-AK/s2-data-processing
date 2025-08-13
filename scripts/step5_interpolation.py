#!/usr/bin/env python3
"""
Step 5: OSRM Interpolation
Processes regional CSV files through OSRM for coordinate snapping, interpolation,
distance calculation, and trip segmentation.

Author: SeeSense Data Pipeline
"""

import pandas as pd
import numpy as np
import os
import sys
import requests
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import polyline
from geopy.distance import geodesic
from tqdm import tqdm
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_manager import ConfigManager
from scripts.utils.logger_setup import setup_logger
from scripts.utils.aws_helper import AWSHelper


class OSRMInterpolator:
    """Handles OSRM-based interpolation and distance calculation for regional data."""
    
    def __init__(self, config_path=None):
        """Initialize the OSRMInterpolator with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logger('osrm_interpolator', self.config.get_log_config())
        
        # Set up directories
        self.base_dir = Path(self.config.get('directories.base_dir', str(project_root)))
        self.preprocessed_dir = self.base_dir / self.config.get('directories.preprocessed_dir', 'data/preprocessed')
        self.processed_dir = self.base_dir / self.config.get('directories.processed_dir', 'data/processed')
        
        # OSRM configuration
        self.osrm_servers = self.config.get_osrm_servers()
        self.trip_break_threshold = self.config.get('processing.trip_break_threshold_minutes', 30) * 60  # Convert to seconds
        
        # Processing settings
        self.max_speed_threshold = self.config.get('processing.max_speed_threshold_kmh', 200)
        
        # Ensure processed directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create region directories in processed folder
        for region in self.config.get_all_regions():
            region_dir = self.processed_dir / region
            region_dir.mkdir(parents=True, exist_ok=True)
    
    def get_yesterday(self):
        """Get yesterday's date in YYYY-MM-DD format."""
        return (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    def get_today(self):
        """Get today's date in YYYY-MM-DD format for testing."""
        return datetime.utcnow().strftime('%Y-%m-%d')
    
    def find_unprocessed_dates(self) -> List[str]:
        """Find dates that have preprocessed data but haven't been processed yet."""
        unprocessed_dates = set()
        
        # Check each region for available preprocessed dates
        for region in self.config.get_all_regions():
            preprocessed_region_dir = self.preprocessed_dir / region
            processed_region_dir = self.processed_dir / region
            
            if not preprocessed_region_dir.exists():
                continue
            
            # Get all date folders in preprocessed
            for date_folder in preprocessed_region_dir.iterdir():
                if not date_folder.is_dir():
                    continue
                
                date_str = date_folder.name
                
                # Check if this date has been processed for this region
                processed_date_dir = processed_region_dir / date_str
                if not processed_date_dir.exists() or not any(processed_date_dir.glob('*.csv')):
                    unprocessed_dates.add(date_str)
        
        return sorted(list(unprocessed_dates))
    
    def get_osrm_url(self, region: str) -> Optional[str]:
        """Get OSRM server URL for a specific region."""
        if region not in self.osrm_servers:
            self.logger.error(f"No OSRM server configured for region: {region}")
            return None
        
        server_config = self.osrm_servers[region]
        host = server_config.get('host', 'localhost')
        port = server_config.get('port')
        
        if not port:
            self.logger.error(f"No port configured for OSRM server in region: {region}")
            return None
        
        return f"http://{host}:{port}"
    
    def test_osrm_connection(self, region: str) -> bool:
        """Test connection to OSRM server for a region."""
        osrm_url = self.get_osrm_url(region)
        if not osrm_url:
            return False
        
        try:
            # Test with a simple nearest query
            test_url = f"{osrm_url}/nearest/v1/bike/0,0"
            response = requests.get(test_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"OSRM connection test failed for {region}: {e}")
            return False
    
    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points in meters."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * 6371000 * np.arcsin(np.sqrt(a))
    
    def snap_coordinate_to_road(self, lat: float, lon: float, osrm_url: str) -> Optional[Tuple[float, float]]:
        """Snap a coordinate to the nearest road using OSRM."""
        try:
            url = f"{osrm_url}/nearest/v1/bike/{lon},{lat}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data['code'] == 'Ok' and data['waypoints']:
                snapped_location = data['waypoints'][0]['location']
                return (snapped_location[1], snapped_location[0])  # Return as (lat, lon)
            
        except Exception as e:
            self.logger.warning(f"Failed to snap coordinate ({lat}, {lon}): {e}")
        
        return None
    
    def get_route_geometry(self, p1: Tuple[float, float], p2: Tuple[float, float], osrm_url: str) -> Optional[List[Tuple[float, float]]]:
        """Get route geometry between two points."""
        try:
            url = f"{osrm_url}/route/v1/bike/{p1[1]},{p1[0]};{p2[1]},{p2[0]}?overview=full&geometries=polyline"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data['code'] == 'Ok' and data['routes']:
                geometry = data['routes'][0]['geometry']
                return polyline.decode(geometry)  # Returns list of (lat, lon) points
                
        except Exception as e:
            self.logger.warning(f"Failed to get route geometry between {p1} and {p2}: {e}")
        
        return None
    
    def calculate_osrm_distance(self, lat1: float, lon1: float, lat2: float, lon2: float, osrm_url: str) -> Optional[float]:
        """Calculate distance between two points using OSRM routing."""
        try:
            url = f"{osrm_url}/route/v1/bike/{lon1},{lat1};{lon2},{lat2}?overview=false"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data['code'] == 'Ok' and data['routes']:
                return data['routes'][0]['distance']  # Distance in meters
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate OSRM distance: {e}")
        
        # Fallback to haversine distance
        return self.haversine(lat1, lon1, lat2, lon2)
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataframe: filter, sort, and prepare for interpolation."""
        initial_count = len(df)
        
        # Step 2: Remove rows with non-null device_serial_number
        if 'device_serial_number' in df.columns:
            df = df[df['device_serial_number'].isna()].copy()
            filtered_count = len(df)
            self.logger.info(f"Filtered out {initial_count - filtered_count} rows with device_serial_number, keeping {filtered_count} rows")
        else:
            self.logger.warning("'device_serial_number' column not found - processing all data")
        
        if len(df) == 0:
            self.logger.warning("No data remaining after filtering")
            return df
        
        # Convert numeric columns
        numeric_columns = ['timestamp', 'position_timestamp', 'record_seqnum', 'position_latitude', 'position_longitude']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Step 3: Sort by device_id, timestamp, position_timestamp, record_seqnum
        sort_columns = ['device_id']
        for col in ['timestamp', 'position_timestamp', 'record_seqnum']:
            if col in df.columns:
                sort_columns.append(col)
        
        df = df.sort_values(by=sort_columns).reset_index(drop=True)
        self.logger.info(f"Sorted data by: {sort_columns}")
        
        # Initialize new columns
        df['snapped_lat'] = pd.NA
        df['snapped_lon'] = pd.NA
        df['was_snapped'] = False
        df['distance_m'] = pd.NA
        df['time_s'] = pd.NA
        df['trip_break'] = False
        
        return df
    
    def snap_coordinates(self, df: pd.DataFrame, osrm_url: str) -> pd.DataFrame:
        """Snap coordinates to roads using OSRM."""
        self.logger.info("Starting coordinate snapping...")
        
        # Find rows with valid coordinates
        valid_coords = df[
            df['position_latitude'].notna() & 
            df['position_longitude'].notna() &
            (df['position_latitude'] != 0) &
            (df['position_longitude'] != 0)
        ].copy()
        
        if valid_coords.empty:
            self.logger.warning("No valid coordinates found for snapping")
            return df
        
        # Remove duplicate coordinates to reduce API calls
        unique_coords = valid_coords.drop_duplicates(subset=['position_latitude', 'position_longitude'])
        self.logger.info(f"Snapping {len(unique_coords)} unique coordinates out of {len(valid_coords)} total valid coordinates")
        
        # Snap unique coordinates
        snapped_coords = {}
        
        for idx, row in tqdm(unique_coords.iterrows(), total=len(unique_coords), desc="Snapping coordinates"):
            lat, lon = row['position_latitude'], row['position_longitude']
            coord_key = (lat, lon)
            
            snapped = self.snap_coordinate_to_road(lat, lon, osrm_url)
            if snapped:
                snapped_coords[coord_key] = snapped
                time.sleep(0.05)  # Rate limiting
            else:
                snapped_coords[coord_key] = (lat, lon)  # Keep original if snapping fails
        
        # Apply snapped coordinates back to dataframe
        for idx, row in df.iterrows():
            if pd.notna(row['position_latitude']) and pd.notna(row['position_longitude']):
                coord_key = (row['position_latitude'], row['position_longitude'])
                if coord_key in snapped_coords:
                    snapped_lat, snapped_lon = snapped_coords[coord_key]
                    df.at[idx, 'snapped_lat'] = snapped_lat
                    df.at[idx, 'snapped_lon'] = snapped_lon
                    df.at[idx, 'was_snapped'] = True
        
        snapped_count = df['snapped_lat'].notna().sum()
        self.logger.info(f"Successfully snapped {snapped_count} coordinates")
        
        return df
    
    def interpolate_missing_points(self, df: pd.DataFrame, osrm_url: str) -> pd.DataFrame:
        """Interpolate missing coordinates using OSRM routing."""
        self.logger.info("Starting interpolation of missing points...")
        
        # Process each device separately
        for device_id in df['device_id'].unique():
            if pd.isna(device_id):
                continue
            
            device_df = df[df['device_id'] == device_id].copy()
            device_indices = device_df.index.tolist()
            
            # Find indices with valid snapped coordinates
            valid_indices = device_df[device_df['snapped_lat'].notna()].index.tolist()
            
            if len(valid_indices) < 2:
                continue  # Need at least 2 points for interpolation
            
            # Process gaps between consecutive valid points
            for i in range(len(valid_indices) - 1):
                start_idx = valid_indices[i]
                end_idx = valid_indices[i + 1]
                
                # Get indices of missing points between start and end
                missing_indices = [idx for idx in device_indices if start_idx < idx < end_idx]
                
                if not missing_indices:
                    continue
                
                # Get route geometry between start and end points
                start_point = (df.at[start_idx, 'snapped_lat'], df.at[start_idx, 'snapped_lon'])
                end_point = (df.at[end_idx, 'snapped_lat'], df.at[end_idx, 'snapped_lon'])
                
                route_coords = self.get_route_geometry(start_point, end_point, osrm_url)
                if not route_coords or len(route_coords) < 2:
                    continue
                
                # Calculate cumulative distances along route
                route_distances = [0]
                for j in range(1, len(route_coords)):
                    dist = geodesic(route_coords[j-1], route_coords[j]).meters
                    route_distances.append(route_distances[-1] + dist)
                
                total_route_distance = route_distances[-1]
                if total_route_distance == 0:
                    continue
                
                # Interpolate missing points along the route
                progress_distance = 0
                
                for missing_idx in missing_indices:
                    # Calculate expected progress based on speed and time
                    time_diff = df.at[missing_idx, 'timestamp'] - df.at[start_idx, 'timestamp']
                    if pd.notna(df.at[missing_idx, 'position_speed']):
                        speed_mps = df.at[missing_idx, 'position_speed'] / 3.6  # km/h to m/s
                        progress_distance += speed_mps * max(1, time_diff / len(missing_indices))
                    else:
                        # Linear interpolation if no speed data
                        position_ratio = (missing_idx - start_idx) / (end_idx - start_idx)
                        progress_distance = total_route_distance * position_ratio
                    
                    # Clamp progress to route bounds
                    progress_distance = max(0, min(progress_distance, total_route_distance))
                    
                    # Find closest point on route
                    closest_route_idx = np.argmin(np.abs(np.array(route_distances) - progress_distance))
                    interpolated_point = route_coords[closest_route_idx]
                    
                    # Snap interpolated point to road
                    snapped_point = self.snap_coordinate_to_road(interpolated_point[0], interpolated_point[1], osrm_url)
                    if snapped_point:
                        df.at[missing_idx, 'snapped_lat'] = snapped_point[0]
                        df.at[missing_idx, 'snapped_lon'] = snapped_point[1]
                        df.at[missing_idx, 'was_snapped'] = True
                
                time.sleep(0.1)  # Rate limiting for route requests
        
        interpolated_count = df['snapped_lat'].notna().sum()
        self.logger.info(f"Interpolation complete. Total points with coordinates: {interpolated_count}")
        
        return df
    
    def calculate_distances_and_times(self, df: pd.DataFrame, osrm_url: str) -> pd.DataFrame:
        """Calculate distances and time differences between consecutive points."""
        self.logger.info("Calculating distances and time differences...")
        
        # Process each device separately
        for device_id in df['device_id'].unique():
            if pd.isna(device_id):
                continue
            
            device_mask = df['device_id'] == device_id
            device_indices = df[device_mask].index.tolist()
            
            if len(device_indices) < 2:
                continue
            
            # Set first point distance and time to 0
            first_idx = device_indices[0]
            df.at[first_idx, 'distance_m'] = 0.0
            df.at[first_idx, 'time_s'] = 0.0
            
            # Calculate for subsequent points
            for i in range(1, len(device_indices)):
                current_idx = device_indices[i]
                previous_idx = device_indices[i-1]
                
                # Calculate time difference
                current_time = df.at[current_idx, 'timestamp']
                previous_time = df.at[previous_idx, 'timestamp']
                
                if pd.notna(current_time) and pd.notna(previous_time):
                    time_diff = current_time - previous_time
                    
                    # Mark trip breaks (Step 6: > 600 seconds = -1)
                    if time_diff > self.trip_break_threshold:
                        df.at[current_idx, 'time_s'] = -1
                        df.at[current_idx, 'trip_break'] = True
                        df.at[current_idx, 'distance_m'] = 0.0  # Reset distance at trip break
                        continue
                    else:
                        df.at[current_idx, 'time_s'] = time_diff
                
                # Calculate distance if we have coordinates
                current_lat = df.at[current_idx, 'snapped_lat']
                current_lon = df.at[current_idx, 'snapped_lon']
                previous_lat = df.at[previous_idx, 'snapped_lat']
                previous_lon = df.at[previous_idx, 'snapped_lon']
                
                if all(pd.notna([current_lat, current_lon, previous_lat, previous_lon])):
                    # Use OSRM for distance calculation
                    distance = self.calculate_osrm_distance(
                        previous_lat, previous_lon, current_lat, current_lon, osrm_url
                    )
                    df.at[current_idx, 'distance_m'] = distance
                else:
                    df.at[current_idx, 'distance_m'] = 0.0
        
        self.logger.info("Distance and time calculations complete")
        return df
    
    def process_region_file(self, file_path: Path, region: str, date_str: str) -> bool:
        """Process a single regional CSV file through OSRM interpolation."""
        try:
            self.logger.info(f"Processing {region} file: {file_path}")
            
            # Test OSRM connection
            if not self.test_osrm_connection(region):
                self.logger.error(f"OSRM server not available for region: {region}")
                return False
            
            osrm_url = self.get_osrm_url(region)
            
            # Load CSV file
            df = pd.read_csv(file_path)
            initial_rows = len(df)
            self.logger.info(f"Loaded {initial_rows} rows from {file_path}")
            
            if initial_rows == 0:
                self.logger.warning(f"Empty CSV file: {file_path}")
                return False
            
            # Step 1: Preprocess data
            df = self.preprocess_dataframe(df)
            
            if len(df) == 0:
                self.logger.warning(f"No data remaining after preprocessing for {file_path}")
                return False
            
            # Step 4: Snap coordinates to roads
            df = self.snap_coordinates(df, osrm_url)
            
            # Step 4: Interpolate missing points
            df = self.interpolate_missing_points(df, osrm_url)
            
            # Steps 5 & 6: Calculate distances and times
            df = self.calculate_distances_and_times(df, osrm_url)
            
            # Save processed file
            output_dir = self.processed_dir / region / date_str
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{region}_{date_str.replace('-', '')}_processed.csv"
            output_path = output_dir / output_filename
            
            df.to_csv(output_path, index=False)
            
            # Log summary
            snapped_count = df['was_snapped'].sum()
            trip_breaks = df['trip_break'].sum()
            valid_distances = df['distance_m'].notna().sum()
            
            self.logger.info(f"✅ Processed {region} for {date_str}:")
            self.logger.info(f"   - Output: {output_path}")
            self.logger.info(f"   - Rows processed: {len(df)}")
            self.logger.info(f"   - Coordinates snapped: {snapped_count}")
            self.logger.info(f"   - Trip breaks detected: {trip_breaks}")
            self.logger.info(f"   - Valid distances calculated: {valid_distances}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {region} file {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def process_date(self, date_str: str, specific_regions: List[str] = None) -> bool:
        """Process all regions for a specific date."""
        try:
            self.logger.info(f"Starting OSRM interpolation for date: {date_str}")
            
            regions_to_process = specific_regions if specific_regions else self.config.get_all_regions()
            successful_regions = []
            failed_regions = []
            
            for region in regions_to_process:
                self.logger.info(f"Processing region: {region}")
                
                # Check if preprocessed file exists
                preprocessed_file_path = self.preprocessed_dir / region / date_str / f"{region}_{date_str.replace('-', '')}.csv"
                
                if not preprocessed_file_path.exists():
                    self.logger.info(f"No preprocessed file found for {region} on {date_str}: {preprocessed_file_path}")
                    continue
                
                # Check if already processed
                processed_dir_path = self.processed_dir / region / date_str
                if processed_dir_path.exists() and any(processed_dir_path.glob('*.csv')):
                    self.logger.info(f"Already processed {region} for {date_str}")
                    successful_regions.append(region)
                    continue
                
                # Process the file
                if self.process_region_file(preprocessed_file_path, region, date_str):
                    successful_regions.append(region)
                else:
                    failed_regions.append(region)
            
            # Summary
            self.logger.info(f"Processing complete for {date_str}:")
            self.logger.info(f"  ✅ Successful regions: {successful_regions}")
            if failed_regions:
                self.logger.warning(f"  ❌ Failed regions: {failed_regions}")
            
            return len(successful_regions) > 0
            
        except Exception as e:
            self.logger.error(f"Error processing date {date_str}: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_interactive(self):
        """Run the interpolator in interactive mode."""
        try:
            # Find unprocessed dates
            unprocessed_dates = self.find_unprocessed_dates()
            
            if not unprocessed_dates:
                print("✅ No unprocessed dates found. All data is up to date!")
                return
            
            print(f"\n📅 Found {len(unprocessed_dates)} unprocessed dates:")
            for i, date in enumerate(unprocessed_dates, 1):
                print(f"  {i}. {date}")
            
            # Let user choose
            print("\nOptions:")
            print("1. Process all unprocessed dates")
            print("2. Process specific date")
            print("3. Process yesterday's data")
            
            choice = input("Select option (1-3) [default: 3]: ").strip()
            
            if choice == '1':
                # Process all unprocessed dates
                for date_str in unprocessed_dates:
                    self.process_date(date_str)
            elif choice == '2':
                # Process specific date
                print("\nAvailable dates:")
                for i, date in enumerate(unprocessed_dates, 1):
                    print(f"  {i}. {date}")
                
                date_choice = input(f"Select date (1-{len(unprocessed_dates)}): ").strip()
                try:
                    date_idx = int(date_choice) - 1
                    if 0 <= date_idx < len(unprocessed_dates):
                        selected_date = unprocessed_dates[date_idx]
                        self.process_date(selected_date)
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Invalid input")
            else:
                # Process yesterday's data (default)
                yesterday = self.get_yesterday()
                if yesterday in unprocessed_dates:
                    self.process_date(yesterday)
                else:
                    print(f"Yesterday's data ({yesterday}) is already processed or not available")
            
        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Unexpected error in interactive mode: {e}")
            sys.exit(1)
    
    def run_automated(self, date_str: str = None) -> bool:
        """Run the interpolator in automated mode (for scheduling)."""
        try:
            if not date_str:
                date_str = self.get_yesterday()
            
            self.logger.info(f"Running automated OSRM interpolation for {date_str}")
            success = self.process_date(date_str)
            
            if success:
                self.logger.info("Automated OSRM interpolation completed successfully")
                return True
            else:
                self.logger.error("Automated OSRM interpolation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in automated processing: {e}")
            self.logger.debug(traceback.format_exc())
            return False


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OSRM Interpolation - Step 5 of S2 Data Pipeline')
    parser.add_argument('--date', type=str, help='Date to process (YYYY-MM-DD). Defaults to yesterday.')
    parser.add_argument('--region', type=str, help='Specific region to process (optional)')
    parser.add_argument('--automated', action='store_true', help='Run in automated mode (no user prompts)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        interpolator = OSRMInterpolator(args.config)
        
        if args.automated:
            success = interpolator.run_automated(args.date)
            sys.exit(0 if success else 1)
        else:
            if args.date:
                regions = [args.region] if args.region else None
                interpolator.process_date(args.date, regions)
            else:
                interpolator.run_interactive()
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()