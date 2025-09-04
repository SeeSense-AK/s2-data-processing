#!/usr/bin/env python3
"""
Step 5: Enhanced OSRM Interpolation
Processes regional CSV files through OSRM for coordinate snapping, interpolation,
distance calculation, and trip segmentation.

Enhanced version that eliminates stale GPS points using advanced interpolation techniques.

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
from scripts.utils.date_utils import normalize_date_format, get_yesterday_formats, get_today_formats


class OSRMInterpolator:
    """Enhanced OSRM-based interpolation that eliminates stale GPS points."""
    
    def __init__(self, config_path=None):
        """Initialize the Enhanced OSRMInterpolator with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logger('enhanced_osrm_interpolator', self.config.get_log_config())
        
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
        """Get yesterday's date in local format."""
        _, local_format, _ = get_yesterday_formats()
        return local_format
    
    def get_today(self):
        """Get today's date in local format."""
        _, local_format, _ = get_today_formats()
        return local_format
    
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
            response = requests.get(url, timeout=10, headers={'Connection': 'close'})
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
        
        # Remove rows with non-null device_serial_number
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
        
        # Sort by device_id, timestamp, position_timestamp, record_seqnum
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
        
        # Calculate time differences per device (grouped)
        if 'timestamp' in df.columns:
            df['time_diff'] = df.groupby('device_id')['timestamp'].diff().fillna(0)
        
        return df
    
    def snap_coordinates_enhanced(self, df: pd.DataFrame, osrm_url: str) -> pd.DataFrame:
        """Enhanced coordinate snapping based on IP-Final approach."""
        self.logger.info("Starting enhanced coordinate snapping...")
        
        # Remove duplicate coordinates to reduce API calls (IP-Final approach)
        # Mark duplicates as NA but preserve first occurrence
        df_coords = df[['position_latitude', 'position_longitude']].copy()
        
        # Find duplicates and mark them as NA (keep='first' preserves the first occurrence)
        duplicate_mask = df_coords.duplicated(subset=['position_latitude', 'position_longitude'], keep='first')
        df.loc[duplicate_mask, ['position_latitude', 'position_longitude']] = pd.NA
        
        # Count duplicates
        num_na = df[['position_latitude', 'position_longitude']].isna().any(axis=1).sum()
        self.logger.info(f"Marked {num_na} duplicate coordinates as NA")
        
        # Find rows with valid coordinates
        valid_coords = df[
            df['position_latitude'].notna() & 
            df['position_longitude'].notna() &
            (df['position_latitude'] != 0) &
            (df['position_longitude'] != 0)
        ]
        
        if valid_coords.empty:
            self.logger.warning("No valid coordinates found for snapping")
            return df
        
        self.logger.info(f"Snapping {len(valid_coords)} unique coordinates")
        
        # Snap coordinates with progress bar
        for idx, row in tqdm(valid_coords.iterrows(), total=len(valid_coords), desc="Snapping coordinates"):
            lat, lon = row['position_latitude'], row['position_longitude']
            
            snapped = self.snap_coordinate_to_road(lat, lon, osrm_url)
            if snapped:
                df.at[idx, 'snapped_lat'] = snapped[0]
                df.at[idx, 'snapped_lon'] = snapped[1]
                df.at[idx, 'was_snapped'] = True
            else:
                # Keep original if snapping fails
                df.at[idx, 'snapped_lat'] = lat
                df.at[idx, 'snapped_lon'] = lon
                df.at[idx, 'was_snapped'] = False
            
            time.sleep(0.05)  # Rate limiting
        
        snapped_count = df['snapped_lat'].notna().sum()
        self.logger.info(f"Successfully processed {snapped_count} coordinates")
        
        return df
    
    def interpolate_gaps_enhanced(self, df: pd.DataFrame, osrm_url: str) -> pd.DataFrame:
        """Enhanced gap interpolation based on IP-Final approach."""
        self.logger.info("Starting enhanced gap interpolation...")
        
        # Find indices of known snapped points
        known_indices = df[df['snapped_lat'].notna()].index.tolist()
        
        if len(known_indices) < 2:
            self.logger.warning("Not enough known points for interpolation")
            return df
        
        # Process each gap between consecutive known points
        for i in tqdm(range(len(known_indices) - 1), desc="Processing gaps"):
            idx1 = known_indices[i]
            idx2 = known_indices[i + 1]
            
            # Get the bounding points
            p1 = (df.at[idx1, 'snapped_lat'], df.at[idx1, 'snapped_lon'])
            p2 = (df.at[idx2, 'snapped_lat'], df.at[idx2, 'snapped_lon'])
            
            # Get the route geometry between points
            route_coords = self.get_route_geometry(p1, p2, osrm_url)
            if not route_coords or len(route_coords) < 2:
                continue
            
            # Calculate cumulative distances along route
            route_dists = [0]
            for j in range(1, len(route_coords)):
                dist = geodesic(route_coords[j-1], route_coords[j]).meters
                route_dists.append(route_dists[-1] + dist)
            
            total_distance = route_dists[-1]
            if total_distance == 0:
                continue
            
            # Process each row in the gap
            progress_meters = 0
            for row_idx in range(idx1 + 1, idx2):
                # Calculate expected progress based on speed and time (IP-Final approach)
                if pd.notna(df.at[row_idx, 'position_speed']) and df.at[row_idx, 'time_diff'] > 0:
                    speed_mps = df.at[row_idx, 'position_speed'] / 3.6  # km/h to m/s
                    time_sec = df.at[row_idx, 'time_diff']
                    progress_meters += speed_mps * time_sec
                else:
                    # Linear interpolation if no speed data
                    position_ratio = (row_idx - idx1) / (idx2 - idx1)
                    progress_meters = total_distance * position_ratio
                
                # Clamp progress to route bounds
                progress_meters = max(0, min(progress_meters, total_distance))
                
                # Find closest point along route
                closest_route_idx = np.argmin(np.abs(np.array(route_dists) - progress_meters))
                interpolated_point = route_coords[closest_route_idx]
                
                # Snap interpolated point to road for accuracy
                snapped_point = self.snap_coordinate_to_road(interpolated_point[0], interpolated_point[1], osrm_url)
                if snapped_point:
                    df.at[row_idx, 'snapped_lat'] = snapped_point[0]
                    df.at[row_idx, 'snapped_lon'] = snapped_point[1]
                    df.at[row_idx, 'was_snapped'] = True
        
        interpolated_count = df['snapped_lat'].notna().sum()
        self.logger.info(f"After gap interpolation: {interpolated_count} points have coordinates")
        
        return df
    
    def fill_missing_coordinates_comprehensive(self, df: pd.DataFrame, osrm_url: str) -> pd.DataFrame:
        """Comprehensive coordinate filling based on IP-Final approach."""
        self.logger.info("Starting comprehensive coordinate filling...")
        
        original_na_count = df['snapped_lat'].isna().sum()
        
        # Find rows with valid coordinates
        valid_coords = df[df['snapped_lat'].notna() & df['snapped_lon'].notna()]
        
        if valid_coords.empty:
            self.logger.warning("No valid coordinates found to use as reference")
            return df
        
        # Track the last valid position
        last_valid_lat = None
        last_valid_lon = None
        
        # Process each row with progress tracking
        for i in tqdm(range(len(df)), desc="Filling gaps"):
            if pd.notna(df.at[i, 'snapped_lat']) and pd.notna(df.at[i, 'snapped_lon']):
                last_valid_lat = df.at[i, 'snapped_lat']
                last_valid_lon = df.at[i, 'snapped_lon']
                continue
            
            if last_valid_lat is None or last_valid_lon is None:
                continue
            
            # Find next valid point
            next_valid_idx = None
            for j in range(i + 1, len(df)):
                if pd.notna(df.at[j, 'snapped_lat']) and pd.notna(df.at[j, 'snapped_lon']):
                    next_valid_idx = j
                    break
            
            if next_valid_idx is None:
                continue
            
            # Get route between points
            try:
                coordinates = f"{last_valid_lon},{last_valid_lat};{df.at[next_valid_idx, 'snapped_lon']},{df.at[next_valid_idx, 'snapped_lat']}"
                url = f"{osrm_url}/route/v1/bike/{coordinates}?overview=full&geometries=polyline&alternatives=false&continue_straight=false"
                
                response = requests.get(url, timeout=10)
                data = response.json()
                
                if 'routes' in data and len(data['routes']) > 0:
                    route_geometry = data['routes'][0]['geometry']
                    route_coords = polyline.decode(route_geometry)
                    # Convert to (lon, lat) for consistent indexing
                    route_coords = [(lon, lat) for lat, lon in route_coords]
                    
                    num_missing = next_valid_idx - i
                    
                    if len(route_coords) >= 2:
                        selected_points = []
                        if num_missing <= len(route_coords) - 2:
                            # Evenly distribute points along route
                            indices = np.linspace(1, len(route_coords) - 2, num_missing, dtype=int)
                            selected_points = [route_coords[idx] for idx in indices]
                        else:
                            # Use all intermediate points and repeat last one if needed
                            for idx in range(1, len(route_coords) - 1):
                                selected_points.append(route_coords[idx])
                            for _ in range(num_missing - len(selected_points)):
                                selected_points.append(route_coords[-2])
                        
                        # Assign interpolated points
                        for j, point in zip(range(i, next_valid_idx), selected_points):
                            df.at[j, 'snapped_lon'] = point[0]
                            df.at[j, 'snapped_lat'] = point[1]
                            df.at[j, 'was_snapped'] = True
                            
            except Exception as e:
                self.logger.warning(f"Error getting route for gap filling: {e}")
                continue
        
        final_na_count = df['snapped_lat'].isna().sum()
        filled_count = original_na_count - final_na_count
        
        self.logger.info(f"Comprehensive filling complete:")
        self.logger.info(f"  - Original NA values: {original_na_count}")
        self.logger.info(f"  - Remaining NA values: {final_na_count}")
        self.logger.info(f"  - Coordinates filled: {filled_count}")
        
        return df
    
    def calculate_distances_and_times_enhanced(self, df: pd.DataFrame, osrm_url: str) -> pd.DataFrame:
        """Enhanced distance and time calculation with trip break detection."""
        self.logger.info("Calculating distances and time differences with trip break detection...")
        
        # Process each device separately
        for device_id in df['device_id'].unique():
            if pd.isna(device_id):
                continue
            
            device_mask = df['device_id'] == device_id
            device_indices = df[device_mask].index.tolist()
            
            if len(device_indices) < 2:
                continue
            
            self.logger.debug(f"Processing device {device_id} with {len(device_indices)} records")
            
            # Set first point distance and time to 0
            first_idx = device_indices[0]
            df.at[first_idx, 'distance_m'] = 0.0
            df.at[first_idx, 'time_s'] = 0.0
            
            # Track if this is the first valid GPS point for trip break logic
            first_valid_gps = True
            
            # Calculate for subsequent points
            for i in range(1, len(device_indices)):
                current_idx = device_indices[i]
                previous_idx = device_indices[i-1]
                
                # Calculate time difference
                current_time = df.at[current_idx, 'timestamp']
                previous_time = df.at[previous_idx, 'timestamp']
                
                if pd.notna(current_time) and pd.notna(previous_time):
                    time_diff = current_time - previous_time
                    
                    # Mark trip breaks (> threshold seconds)
                    if time_diff > self.trip_break_threshold:
                        df.at[current_idx, 'time_s'] = -1  # Trip break marker
                        df.at[current_idx, 'trip_break'] = True
                        df.at[current_idx, 'distance_m'] = 0.0  # Reset distance at trip break
                        first_valid_gps = True  # Reset for new trip
                        continue
                    else:
                        df.at[current_idx, 'time_s'] = time_diff
                
                # Handle first valid GPS point logic with safer NA checking
                current_lat_valid = pd.notna(df.at[current_idx, 'snapped_lat'])
                current_lon_valid = pd.notna(df.at[current_idx, 'snapped_lon'])
                
                if first_valid_gps and current_lat_valid and current_lon_valid:
                    df.at[current_idx, 'distance_m'] = 0.0
                    first_valid_gps = False
                    continue
                
                # Calculate distance if we have coordinates for both points
                current_lat = df.at[current_idx, 'snapped_lat']
                current_lon = df.at[current_idx, 'snapped_lon']
                previous_lat = df.at[previous_idx, 'snapped_lat']
                previous_lon = df.at[previous_idx, 'snapped_lon']
                
                # Safer NA checking to avoid boolean ambiguity
                coords_valid = (pd.notna(current_lat) and pd.notna(current_lon) and 
                              pd.notna(previous_lat) and pd.notna(previous_lon))
                
                if coords_valid:
                    # Use OSRM for accurate distance calculation
                    distance = self.calculate_osrm_distance(
                        previous_lat, previous_lon, current_lat, current_lon, osrm_url
                    )
                    df.at[current_idx, 'distance_m'] = distance if distance is not None else 0.0
                else:
                    df.at[current_idx, 'distance_m'] = 0.0
        
        # Update speed calculations based on new distances (like in IP-Final)
        for i in range(1, len(df)):
            # Safer boolean checking for NA values
            current_lat_valid = pd.notna(df.at[i, 'snapped_lat'])
            previous_lat_valid = pd.notna(df.at[i-1, 'snapped_lat'])
            time_s_value = df.at[i, 'time_s']
            time_s_valid = pd.notna(time_s_value) and time_s_value > 0
            
            if current_lat_valid and previous_lat_valid and time_s_valid:
                distance = df.at[i, 'distance_m']
                time_sec = df.at[i, 'time_s']
                
                # Additional safety checks
                if (pd.notna(distance) and pd.notna(time_sec) and 
                    time_sec > 0 and distance > 0):
                    # Calculate speed in km/h
                    speed_kmh = (distance / time_sec) * 3.6
                    # Apply speed threshold filter
                    if speed_kmh <= self.max_speed_threshold:
                        df.at[i, 'position_speed'] = speed_kmh
                    else:
                        df.at[i, 'position_speed'] = 0
                else:
                    df.at[i, 'position_speed'] = 0
        
        # Safe counting with NA handling
        trip_breaks = df['trip_break'].fillna(False).sum()
        valid_distances = df['distance_m'].notna().sum()
        
        self.logger.info(f"Distance and time calculations complete:")
        self.logger.info(f"  - Trip breaks detected: {trip_breaks}")
        self.logger.info(f"  - Valid distances calculated: {valid_distances}")
        
        return df
    
    def process_region_file(self, file_path: Path, region: str, date_str: str) -> bool:
        """Process a single regional CSV file through enhanced OSRM interpolation."""
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
            
            # Step 2: Enhanced coordinate snapping (removes duplicates)
            df = self.snap_coordinates_enhanced(df, osrm_url)
            
            # Step 3: Enhanced gap interpolation between known points
            df = self.interpolate_gaps_enhanced(df, osrm_url)
            
            # Step 4: Comprehensive coordinate filling for remaining gaps
            df = self.fill_missing_coordinates_comprehensive(df, osrm_url)
            
            # Step 5: Enhanced distance and time calculations with trip breaks
            df = self.calculate_distances_and_times_enhanced(df, osrm_url)
            
            # Save processed file using utility function for consistent naming
            _, date_folder, date_compact = normalize_date_format(date_str)
            output_dir = self.processed_dir / region / date_folder
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{region}_{date_compact}_processed.csv"
            output_path = output_dir / output_filename
            
            df.to_csv(output_path, index=False)
            
            # Log comprehensive summary
            snapped_count = df['was_snapped'].sum()
            trip_breaks = df['trip_break'].sum()
            valid_distances = df['distance_m'].notna().sum()
            remaining_na = df['snapped_lat'].isna().sum()
            total_coordinates = len(df)
            filled_percentage = ((total_coordinates - remaining_na) / total_coordinates) * 100
            
            self.logger.info(f"✅ Enhanced processing complete for {region} on {date_str}:")
            self.logger.info(f"   - Output: {output_path}")
            self.logger.info(f"   - Total rows: {len(df)}")
            self.logger.info(f"   - Coordinates filled: {filled_percentage:.1f}% ({total_coordinates - remaining_na}/{total_coordinates})")
            self.logger.info(f"   - Coordinates snapped: {snapped_count}")
            self.logger.info(f"   - Trip breaks detected: {trip_breaks}")
            self.logger.info(f"   - Valid distances: {valid_distances}")
            self.logger.info(f"   - Remaining stale GPS points: {remaining_na}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {region} file {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def process_date(self, date_str: str, specific_regions: List[str] = None) -> bool:
        """Process all regions for a specific date using enhanced interpolation."""
        try:
            self.logger.info(f"Starting enhanced OSRM interpolation for date: {date_str}")
            
            # Use utility function to normalize date format
            _, date_folder, date_compact = normalize_date_format(date_str)
            
            regions_to_process = specific_regions if specific_regions else self.config.get_all_regions()
            successful_regions = []
            failed_regions = []
            
            for region in regions_to_process:
                self.logger.info(f"Processing region: {region}")
                
                # Check if preprocessed file exists using consistent naming
                preprocessed_file_path = self.preprocessed_dir / region / date_folder / f"{region}_{date_compact}.csv"
                
                if not preprocessed_file_path.exists():
                    self.logger.info(f"No preprocessed file found for {region} on {date_str}: {preprocessed_file_path}")
                    continue
                
                # Check if already processed
                processed_dir_path = self.processed_dir / region / date_folder
                if processed_dir_path.exists() and any(processed_dir_path.glob('*.csv')):
                    self.logger.info(f"Already processed {region} for {date_str}")
                    successful_regions.append(region)
                    continue
                
                # Process the file with enhanced interpolation
                if self.process_region_file(preprocessed_file_path, region, date_str):
                    successful_regions.append(region)
                else:
                    failed_regions.append(region)
            
            # Summary
            self.logger.info(f"Enhanced processing complete for {date_str}:")
            self.logger.info(f"  ✅ Successful regions: {successful_regions}")
            if failed_regions:
                self.logger.warning(f"  ❌ Failed regions: {failed_regions}")
            
            return len(successful_regions) > 0
            
        except Exception as e:
            self.logger.error(f"Error processing date {date_str}: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_interactive(self):
        """Run the enhanced interpolator in interactive mode."""
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
                    print(f"\n🔄 Processing {date_str}...")
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
                        print(f"\n🔄 Processing {selected_date}...")
                        self.process_date(selected_date)
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Invalid input")
            else:
                # Process yesterday's data (default)
                yesterday = self.get_yesterday()
                if yesterday in unprocessed_dates:
                    print(f"\n🔄 Processing yesterday's data: {yesterday}...")
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
        """Run the enhanced interpolator in automated mode (for scheduling)."""
        try:
            if not date_str:
                date_str = self.get_yesterday()
            
            self.logger.info(f"Running automated enhanced OSRM interpolation for {date_str}")
            success = self.process_date(date_str)
            
            if success:
                self.logger.info("Automated enhanced OSRM interpolation completed successfully")
                return True
            else:
                self.logger.error("Automated enhanced OSRM interpolation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in automated processing: {e}")
            self.logger.debug(traceback.format_exc())
            return False


def main():
    """Main entry point for the enhanced interpolation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced OSRM Interpolation - Step 5 of S2 Data Pipeline')
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
                print(f"🚀 Starting enhanced interpolation for {args.date}")
                interpolator.process_date(args.date, regions)
                print("✅ Enhanced interpolation completed!")
            else:
                interpolator.run_interactive()
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
