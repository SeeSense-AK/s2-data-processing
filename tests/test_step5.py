#!/usr/bin/env python3
"""
Test Script for Step 5 - OSRM Interpolation
Tests the OSRM interpolation functionality and server connectivity.

Author: SeeSense Data Pipeline
"""

import sys
import requests
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scripts.utils.config_manager import ConfigManager
from scripts.utils.logger_setup import quick_logger


def test_configuration():
    """Test configuration loading."""
    print("🔧 Testing Configuration...")
    
    try:
        config = ConfigManager()
        config.validate_config()
        print("✅ Configuration loaded and validated successfully")
        
        # Print OSRM server configuration
        osrm_servers = config.get_osrm_servers()
        print(f"   - OSRM Servers configured: {list(osrm_servers.keys())}")
        
        return config
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return None


def test_osrm_servers(config):
    """Test connectivity to all OSRM servers."""
    print("\n🌐 Testing OSRM Server Connectivity...")
    
    osrm_servers = config.get_osrm_servers()
    if not osrm_servers:
        print("❌ No OSRM servers configured")
        return False
    
    all_servers_ok = True
    
    for region, server_config in osrm_servers.items():
        try:
            host = server_config.get('host', 'localhost')
            port = server_config.get('port')
            container_name = server_config.get('container_name', 'unknown')
            
            if not port:
                print(f"❌ {region}: No port configured")
                all_servers_ok = False
                continue
            
            # Test server connectivity
            test_url = f"http://{host}:{port}/nearest/v1/bike/0,0"
            response = requests.get(test_url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ {region}: Server running on port {port} (container: {container_name})")
            else:
                print(f"❌ {region}: Server responded with status {response.status_code}")
                all_servers_ok = False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {region}: Connection refused on port {port} (container: {container_name})")
            all_servers_ok = False
        except requests.exceptions.Timeout:
            print(f"❌ {region}: Connection timeout on port {port}")
            all_servers_ok = False
        except Exception as e:
            print(f"❌ {region}: Error testing server - {e}")
            all_servers_ok = False
    
    return all_servers_ok


def test_directory_structure(config):
    """Test directory structure and find available data."""
    print("\n📁 Testing Directory Structure...")
    
    try:
        base_dir = Path(config.get('directories.base_dir', str(project_root)))
        preprocessed_dir = base_dir / config.get('directories.preprocessed_dir', 'data/preprocessed')
        processed_dir = base_dir / config.get('directories.processed_dir', 'data/processed')
        
        print(f"Base directory: {base_dir}")
        print(f"Preprocessed directory: {preprocessed_dir}")
        print(f"Processed directory: {processed_dir}")
        
        # Check if directories exist
        if preprocessed_dir.exists():
            print("✅ Preprocessed directory exists")
            
            # Check for available data
            available_data = {}
            for region_dir in preprocessed_dir.iterdir():
                if region_dir.is_dir():
                    region = region_dir.name
                    dates = []
                    for date_dir in region_dir.iterdir():
                        if date_dir.is_dir():
                            csv_files = list(date_dir.glob('*.csv'))
                            if csv_files:
                                dates.append(date_dir.name)
                    if dates:
                        available_data[region] = sorted(dates)
            
            if available_data:
                print("📊 Available preprocessed data:")
                for region, dates in available_data.items():
                    print(f"   - {region}: {dates}")
                return available_data
            else:
                print("❌ No preprocessed data found")
                return {}
        else:
            print("❌ Preprocessed directory does not exist")
            return {}
            
    except Exception as e:
        print(f"❌ Error checking directories: {e}")
        return {}


def test_sample_interpolation(config, available_data):
    """Test interpolation on a small sample."""
    print("\n🧪 Testing Sample Interpolation...")
    
    if not available_data:
        print("❌ No data available for testing")
        return False
    
    # Find a region and date with data
    test_region = None
    test_date = None
    
    for region, dates in available_data.items():
        if dates:
            test_region = region
            test_date = dates[0]  # Use first available date
            break
    
    if not test_region or not test_date:
        print("❌ No suitable test data found")
        return False
    
    try:
        # Import the interpolator
        from scripts.step5_interpolation import OSRMInterpolator
        
        interpolator = OSRMInterpolator()
        
        # Test OSRM connection for this region
        if not interpolator.test_osrm_connection(test_region):
            print(f"❌ OSRM server not available for {test_region}")
            return False
        
        print(f"🎯 Testing with region: {test_region}, date: {test_date}")
        
        # Find the test file
        base_dir = Path(config.get('directories.base_dir', str(project_root)))
        preprocessed_dir = base_dir / config.get('directories.preprocessed_dir', 'data/preprocessed')
        test_file = preprocessed_dir / test_region / test_date / f"{test_region}_{test_date.replace('-', '')}.csv"
        
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            return False
        
        print(f"📄 Using test file: {test_file}")
        
        # Read a small sample for testing
        import pandas as pd
        df = pd.read_csv(test_file)
        
        # Take first 50 rows for quick testing
        sample_df = df.head(50)
        print(f"📊 Sample data: {len(sample_df)} rows")
        print(f"   - Columns: {list(sample_df.columns)}")
        print(f"   - Devices: {sample_df['device_id'].nunique() if 'device_id' in sample_df.columns else 'N/A'}")
        
        # Test coordinate snapping on a few points
        osrm_url = interpolator.get_osrm_url(test_region)
        
        # Find first valid coordinate
        valid_coords = sample_df[
            sample_df['position_latitude'].notna() & 
            sample_df['position_longitude'].notna() &
            (sample_df['position_latitude'] != 0) &
            (sample_df['position_longitude'] != 0)
        ]
        
        if len(valid_coords) > 0:
            test_row = valid_coords.iloc[0]
            test_lat = test_row['position_latitude']
            test_lon = test_row['position_longitude']
            
            print(f"🎯 Testing coordinate snapping: ({test_lat}, {test_lon})")
            
            snapped = interpolator.snap_coordinate_to_road(test_lat, test_lon, osrm_url)
            if snapped:
                print(f"✅ Coordinate snapped successfully: ({snapped[0]:.6f}, {snapped[1]:.6f})")
                
                # Test distance calculation
                distance = interpolator.calculate_osrm_distance(test_lat, test_lon, snapped[0], snapped[1], osrm_url)
                if distance is not None:
                    print(f"✅ Distance calculation successful: {distance:.2f} meters")
                else:
                    print("❌ Distance calculation failed")
            else:
                print("❌ Coordinate snapping failed")
                return False
        else:
            print("❌ No valid coordinates found in sample data")
            return False
        
        print("✅ Sample interpolation test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error during sample interpolation test: {e}")
        return False


def test_step5_dry_run(config, available_data):
    """Perform a dry run test of Step 5."""
    print(f"\n🏃 Testing Step 5 (Dry Run)")
    
    if not available_data:
        print("❌ No data available for dry run")
        return False
    
    try:
        # Import the interpolator
        from scripts.step5_interpolation import OSRMInterpolator
        
        interpolator = OSRMInterpolator()
        
        # Find unprocessed dates
        unprocessed_dates = interpolator.find_unprocessed_dates()
        
        if unprocessed_dates:
            print(f"📅 Found {len(unprocessed_dates)} unprocessed dates: {unprocessed_dates}")
            
            # Test with first unprocessed date
            test_date = unprocessed_dates[0]
            print(f"🎯 Would process date: {test_date}")
            
            # Check what regions have data for this date
            regions_with_data = []
            for region in config.get_all_regions():
                base_dir = Path(config.get('directories.base_dir', str(project_root)))
                preprocessed_dir = base_dir / config.get('directories.preprocessed_dir', 'data/preprocessed')
                region_file = preprocessed_dir / region / test_date / f"{region}_{test_date.replace('-', '')}.csv"
                
                if region_file.exists():
                    file_size = region_file.stat().st_size / 1024  # KB
                    regions_with_data.append((region, file_size))
            
            if regions_with_data:
                print("📊 Regions with data for this date:")
                for region, size in regions_with_data:
                    osrm_status = "✅" if interpolator.test_osrm_connection(region) else "❌"
                    print(f"   - {region}: {size:.1f} KB {osrm_status}")
            else:
                print("❌ No regional data found for test date")
                return False
            
        else:
            print("✅ All dates are already processed")
        
        return True
        
    except Exception as e:
        print(f"❌ Dry run error: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 S2 Data Pipeline - Step 5 Test Suite")
    print("=" * 50)
    
    # Test 1: Configuration
    config = test_configuration()
    if not config:
        print("\n❌ Configuration test failed. Please check your config files.")
        sys.exit(1)
    
    # Test 2: OSRM Servers
    if not test_osrm_servers(config):
        print("\n❌ OSRM server test failed. Please check your Docker containers.")
        print("\n💡 Troubleshooting tips:")
        print("   - Check if Docker is running: docker ps")
        print("   - Start missing containers: docker start <container-name>")
        print("   - Check container logs: docker logs <container-name>")
        sys.exit(1)
    
    # Test 3: Directory Structure
    available_data = test_directory_structure(config)
    if not available_data:
        print("\n❌ No preprocessed data available for testing.")
        print("\n💡 Make sure you've run Step 4 (Device Bifurcation) first.")
        sys.exit(1)
    
    # Test 4: Sample Interpolation
    if not test_sample_interpolation(config, available_data):
        print("\n❌ Sample interpolation test failed.")
        sys.exit(1)
    
    # Test 5: Dry Run
    if test_step5_dry_run(config, available_data):
        print("\n✅ All tests passed!")
        print("\n🎯 Ready to run Step 5:")
        print("   python scripts/step5_interpolation.py")
        print("   OR")
        print("   python scripts/step5_interpolation.py --date YYYY-MM-DD")
        print("   OR")
        print("   python scripts/step5_interpolation.py --automated  # for scheduled runs")
    else:
        print("\n❌ Dry run failed. Check the logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()