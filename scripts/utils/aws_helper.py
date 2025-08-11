#!/usr/bin/env python3
"""
AWS Helper for S2 Data Pipeline
Handles S3 operations and AWS connectivity.

Author: SeeSense Data Pipeline
"""

import boto3
import os
from botocore.exceptions import ClientError, NoCredentialsError
import logging


class AWSHelper:
    """Helper class for AWS S3 operations."""
    
    def __init__(self, aws_config):
        """Initialize AWS helper with configuration."""
        self.config = aws_config
        self.bucket_name = aws_config.get('bucket_name')
        
        # Initialize S3 client
        self.s3_client = self._create_s3_client()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def _create_s3_client(self):
        """Create and configure S3 client."""
        try:
            # Try using explicit credentials first
            if 'access_key_id' in self.config and 'secret_access_key' in self.config:
                return boto3.client(
                    's3',
                    aws_access_key_id=self.config['access_key_id'],
                    aws_secret_access_key=self.config['secret_access_key'],
                    region_name=self.config.get('region', 'eu-west-1')
                )
            else:
                # Fall back to default credential chain (AWS CLI, environment, IAM role)
                return boto3.client(
                    's3',
                    region_name=self.config.get('region', 'eu-west-1')
                )
                
        except Exception as e:
            self.logger.error(f"Failed to create S3 client: {e}")
            raise
    
    def test_connection(self):
        """Test AWS S3 connectivity."""
        try:
            # Try to list buckets to test connectivity
            self.s3_client.list_buckets()
            return True
        except NoCredentialsError:
            self.logger.error("AWS credentials not found")
            return False
        except ClientError as e:
            self.logger.error(f"AWS connection failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error testing AWS connection: {e}")
            return False
    
    def get_bucket_info(self):
        """Get information about the configured bucket."""
        try:
            # Check if bucket exists and is accessible
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            return {
                'name': self.bucket_name,
                'exists': True,
                'accessible': True
            }
            
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                return {
                    'name': self.bucket_name,
                    'exists': False,
                    'accessible': False,
                    'error': 'Bucket not found'
                }
            elif error_code == 403:
                return {
                    'name': self.bucket_name,
                    'exists': True,
                    'accessible': False,
                    'error': 'Access denied'
                }
            else:
                return {
                    'name': self.bucket_name,
                    'exists': False,
                    'accessible': False,
                    'error': str(e)
                }
    
    def list_files(self, prefix, max_files=None):
        """List files in S3 with given prefix."""
        try:
            files = []
            continuation_token = None
            
            while True:
                if continuation_token:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        Prefix=prefix,
                        ContinuationToken=continuation_token
                    )
                else:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        Prefix=prefix
                    )
                
                if 'Contents' in response:
                    batch_files = [item['Key'] for item in response['Contents']]
                    files.extend(batch_files)
                
                # Check if we've reached max_files limit
                if max_files and len(files) >= max_files:
                    files = files[:max_files]
                    break
                
                # Check if there are more files to fetch
                if not response.get('IsTruncated'):
                    break
                    
                continuation_token = response.get('NextContinuationToken')
            
            return files
            
        except ClientError as e:
            self.logger.error(f"Error listing files with prefix {prefix}: {e}")
            return []
    
    def get_file_info(self, key):
        """Get information about a specific file."""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return {
                'key': key,
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'etag': response.get('ETag')
            }
        except ClientError as e:
            self.logger.error(f"Error getting file info for {key}: {e}")
            return None
    
    def download_file(self, key, local_path):
        """Download a file from S3 to local path."""
        try:
            self.s3_client.download_file(self.bucket_name, key, local_path)
            return True
        except ClientError as e:
            self.logger.error(f"Error downloading {key}: {e}")
            return False
    
    def upload_file(self, local_path, key):
        """Upload a local file to S3."""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, key)
            return True
        except ClientError as e:
            self.logger.error(f"Error uploading {local_path} to {key}: {e}")
            return False
    
    def file_exists(self, key):
        """Check if a file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if int(e.response['Error']['Code']) == 404:
                return False
            else:
                self.logger.error(f"Error checking if file exists {key}: {e}")
                return False
    
    def delete_file(self, key):
        """Delete a file from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            self.logger.error(f"Error deleting {key}: {e}")
            return False
    
    def get_file_size(self, key):
        """Get the size of a file in S3."""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return response.get('ContentLength', 0)
        except ClientError as e:
            self.logger.error(f"Error getting file size for {key}: {e}")
            return 0
