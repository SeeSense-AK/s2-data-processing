#!/usr/bin/env python3
"""
Configuration Manager
Handles loading and accessing configuration from JSON files.

Author: SeeSense Data Pipeline
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        self.project_root = Path(__file__).parent.parent.parent
        
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.project_root / "config" / "pipeline_config.json"
        
        self.aws_config_path = self.project_root / "config" / "aws_config.json"
        self.device_mapping_path = self.project_root / "config" / "device_mapping.json"
        
        self._config = None
        self._aws_config = None
        self._device_mapping = None
        
        # Load configurations
        self._load_config()
        self._load_aws_config()
        self._load_device_mapping()
    
    def _load_config(self):
        """Load main pipeline configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)
            
            # Replace {username} placeholder with actual username
            username = os.getenv('USER') or os.getenv('USERNAME') or 'user'
            self._replace_placeholders(self._config, {'username': username})
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _load_aws_config(self):
        """Load AWS configuration."""
        try:
            if self.aws_config_path.exists():
                with open(self.aws_config_path, 'r') as f:
                    self._aws_config = json.load(f)
            else:
                # Use environment variables if config file doesn't exist
                self._aws_config = {
                    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                    'region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
                }
                
                # Validate that we have credentials
                if not all([self._aws_config['aws_access_key_id'], 
                           self._aws_config['aws_secret_access_key']]):
                    raise ValueError("AWS credentials not found in config file or environment variables")
                    
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in AWS configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading AWS configuration: {e}")
    
    def _load_device_mapping(self):
        """Load device mapping configuration."""
        try:
            if self.device_mapping_path.exists():
                with open(self.device_mapping_path, 'r') as f:
                    self._device_mapping = json.load(f)
            else:
                # Create default empty mapping
                self._device_mapping = {
                    "device_regions": {},
                    "region_info": {}
                }
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in device mapping file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading device mapping: {e}")
    
    def _replace_placeholders(self, obj: Any, replacements: Dict[str, str]):
        """Recursively replace placeholders in configuration."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._replace_placeholders(value, replacements)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._replace_placeholders(item, replacements)
        elif isinstance(obj, str):
            for placeholder, replacement in replacements.items():
                obj = obj.replace(f'{{{placeholder}}}', replacement)
        
        return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation (e.g., 'aws.bucket_name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_aws_config(self) -> Dict[str, Any]:
        """Get AWS configuration."""
        return self._aws_config.copy() if self._aws_config else {}
    
    def get_device_mapping(self) -> Dict[str, Any]:
        """Get device mapping configuration."""
        return self._device_mapping.copy() if self._device_mapping else {}
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            'level': self.get('logging.level', 'INFO'),
            'format': self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'max_file_size_mb': self.get('logging.max_file_size_mb', 10),
            'backup_count': self.get('logging.backup_count', 5),
            'logs_dir': self.get('directories.logs_dir', 'logs')
        }
    
    def get_osrm_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get OSRM server configurations."""
        return self.get('osrm_servers', {})
    
    def get_region_for_device(self, device_name: str) -> Optional[str]:
        """
        Get region for a specific device based on its name prefix.
        
        Args:
            device_name: Device name/identifier
            
        Returns:
            Region name or None if not found
        """
        device_prefixes = self._device_mapping.get('device_prefixes', {})
        
        # Check each region's prefixes
        for region, prefixes in device_prefixes.items():
            for prefix in prefixes:
                if device_name.upper().startswith(prefix.upper()):
                    return region
        
        # Fallback: check old device_regions format for backward compatibility
        device_regions = self._device_mapping.get('device_regions', {})
        for region, devices in device_regions.items():
            if device_name in devices:
                return region
        
        return None
    
    def get_devices_for_region(self, region: str) -> list:
        """
        Get list of device prefixes for a specific region.
        
        Args:
            region: Region name
            
        Returns:
            List of device prefixes for the region
        """
        device_prefixes = self._device_mapping.get('device_prefixes', {})
        return device_prefixes.get(region, [])
    
    def get_all_regions(self) -> list:
        """Get list of all configured regions."""
        device_prefixes = self._device_mapping.get('device_prefixes', {})
        device_regions = self._device_mapping.get('device_regions', {})
        
        # Combine both sources and remove duplicates
        all_regions = list(set(list(device_prefixes.keys()) + list(device_regions.keys())))
        return all_regions
    
    def get_region_info(self, region: str) -> Dict[str, Any]:
        """
        Get information for a specific region.
        
        Args:
            region: Region name
            
        Returns:
            Region information dictionary
        """
        return self._device_mapping.get('region_info', {}).get(region, {})
    
    def get_region_info(self, region: str) -> Dict[str, Any]:
        """
        Get information for a specific region.
        
        Args:
            region: Region name
            
        Returns:
            Region information dictionary
        """
        return self._device_mapping.get('region_info', {}).get(region, {})
    
    def validate_config(self) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = [
            'aws.bucket_name',
            'aws.source_prefix',
            'aws.daily_csv_prefix',
            'directories.base_dir'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        # Validate AWS config
        aws_config = self.get_aws_config()
        if not aws_config.get('aws_access_key_id') or not aws_config.get('aws_secret_access_key'):
            raise ValueError("AWS credentials are required")
        
        return True
    
    def reload(self):
        """Reload all configuration files."""
        self._load_config()
        self._load_aws_config()
        self._load_device_mapping()
    
    def __str__(self) -> str:
        """String representation of the configuration (without sensitive data)."""
        safe_config = self._config.copy() if self._config else {}
        
        # Remove sensitive information
        if 'aws' in safe_config:
            safe_config['aws'] = {k: v for k, v in safe_config['aws'].items() 
                                 if 'key' not in k.lower() and 'secret' not in k.lower()}
        
        return json.dumps(safe_config, indent=2)
