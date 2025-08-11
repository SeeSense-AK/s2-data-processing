#!/usr/bin/env python3
"""
Configuration Manager for S2 Data Pipeline
Handles loading and validation of configuration files.

Author: SeeSense Data Pipeline
"""

import json
import os
from pathlib import Path


class ConfigManager:
    """Manages pipeline configuration from JSON files."""
    
    def __init__(self, config_path=None):
        """Initialize ConfigManager with optional custom config path."""
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to config/pipeline_config.json relative to project root
            project_root = Path(__file__).parent.parent.parent
            self.config_path = project_root / 'config' / 'pipeline_config.json'
        
        self.config = self._load_config()
        self._load_aws_config()
    
    def _load_config(self):
        """Load the main pipeline configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def _load_aws_config(self):
        """Load AWS credentials configuration."""
        aws_config_path = self.config_path.parent / 'aws_config.json'
        
        try:
            with open(aws_config_path, 'r') as f:
                aws_config = json.load(f)
                # Merge AWS config into main config
                self.config['aws'].update(aws_config)
        except FileNotFoundError:
            # Try to get AWS credentials from environment variables
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            if aws_access_key and aws_secret_key:
                self.config['aws'].update({
                    'access_key_id': aws_access_key,
                    'secret_access_key': aws_secret_key
                })
            else:
                print("⚠️  AWS credentials not found in config file or environment variables")
                print("   Make sure to either:")
                print("   1. Create config/aws_config.json with your AWS credentials")
                print("   2. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
                print("   3. Use AWS CLI configured credentials")
    
    def get(self, key, default=None):
        """Get configuration value using dot notation (e.g., 'aws.bucket_name')."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_aws_config(self):
        """Get AWS-specific configuration."""
        return self.config.get('aws', {})
    
    def get_osrm_servers(self):
        """Get OSRM server configurations."""
        return self.config.get('osrm_servers', {})
    
    def get_log_config(self):
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def validate_config(self):
        """Validate that required configuration values are present."""
        required_keys = [
            'aws.bucket_name',
            'aws.source_prefix',
            'aws.daily_csv_prefix',
            'directories.base_dir',
            'directories.download_dir',
            'directories.combined_dir'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        # Validate OSRM servers
        osrm_servers = self.get_osrm_servers()
        if not osrm_servers:
            print("⚠️  No OSRM servers configured")
        
        return True
    
    def get_device_mapping_path(self):
        """Get path to device mapping configuration."""
        return self.config_path.parent / 'device_mapping.json'
    
    def load_device_mapping(self):
        """Load device to region mapping."""
        mapping_path = self.get_device_mapping_path()
        
        try:
            with open(mapping_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Device mapping file not found: {mapping_path}")
            return {}
    
    def __str__(self):
        """String representation of configuration (without sensitive data)."""
        safe_config = self.config.copy()
        
        # Remove sensitive AWS credentials
        if 'aws' in safe_config:
            aws_config = safe_config['aws'].copy()
            for sensitive_key in ['access_key_id', 'secret_access_key']:
                if sensitive_key in aws_config:
                    aws_config[sensitive_key] = '***'
            safe_config['aws'] = aws_config
        
        return json.dumps(safe_config, indent=2)
