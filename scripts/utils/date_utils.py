#!/usr/bin/env python3
"""
Date Utilities for S2 Data Pipeline
Provides consistent date format handling across all pipeline steps.

Author: SeeSense Data Pipeline
"""

from datetime import datetime
from typing import Tuple


def normalize_date_format(date_str: str) -> Tuple[str, str, str]:
    """
    Normalize any date format to standard pipeline formats.
    
    This function handles the three date formats used across the pipeline:
    - AWS format (YYYY/MM/DD) for S3 operations in Step 3
    - Local format (YYYY-MM-DD) for directory names in Steps 4-6
    - Compact format (YYYYMMDD) for filenames across all steps
    
    Args:
        date_str: Date in any supported format:
                 - YYYY/MM/DD (e.g., "2025/08/13")
                 - YYYY-MM-DD (e.g., "2025-08-13") 
                 - YYYYMMDD (e.g., "20250813")
                 
    Returns:
        Tuple[str, str, str]: (aws_format, local_format, compact_format)
        
    Example:
        >>> normalize_date_format("2025/08/13")
        ('2025/08/13', '2025-08-13', '20250813')
        
        >>> normalize_date_format("2025-08-13")  
        ('2025/08/13', '2025-08-13', '20250813')
        
        >>> normalize_date_format("20250813")
        ('2025/08/13', '2025-08-13', '20250813')
        
    Raises:
        ValueError: If date_str is not in a recognized format or is invalid
    """
    if not date_str:
        raise ValueError("Date string cannot be empty")
    
    # Remove any separators to get compact format
    compact = date_str.replace('/', '').replace('-', '')
    
    # Validate compact format
    if len(compact) != 8:
        raise ValueError(f"Invalid date format: '{date_str}'. Expected YYYY/MM/DD, YYYY-MM-DD, or YYYYMMDD")
    
    if not compact.isdigit():
        raise ValueError(f"Invalid date format: '{date_str}'. Date must contain only digits and separators")
    
    # Extract components
    year = compact[:4]
    month = compact[4:6] 
    day = compact[6:8]
    
    # Validate date components
    try:
        # This will raise ValueError if date is invalid (e.g., month 13, day 32)
        datetime.strptime(compact, '%Y%m%d')
    except ValueError as e:
        raise ValueError(f"Invalid date: '{date_str}'. {str(e)}")
    
    # Return all three standard formats
    aws_format = f"{year}/{month}/{day}"      # For Step 3 AWS/S3 operations
    local_format = f"{year}-{month}-{day}"     # For Steps 4-6 local operations  
    compact_format = compact                   # For filenames across all steps
    
    return aws_format, local_format, compact_format


def get_yesterday_formats() -> Tuple[str, str, str]:
    """
    Get yesterday's date in all three standard formats.
    
    Returns:
        Tuple[str, str, str]: (aws_format, local_format, compact_format)
        
    Example:
        >>> get_yesterday_formats()
        ('2025/08/12', '2025-08-12', '20250812')
    """
    from datetime import timedelta
    yesterday = datetime.utcnow() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    return normalize_date_format(yesterday_str)


def get_today_formats() -> Tuple[str, str, str]:
    """
    Get today's date in all three standard formats.
    
    Returns:
        Tuple[str, str, str]: (aws_format, local_format, compact_format)
        
    Example:
        >>> get_today_formats()
        ('2025/08/13', '2025-08-13', '20250813')
    """
    today = datetime.utcnow()
    today_str = today.strftime('%Y-%m-%d') 
    return normalize_date_format(today_str)


def validate_date_format(date_str: str) -> bool:
    """
    Validate if a date string is in a supported format.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        bool: True if format is valid, False otherwise
        
    Example:
        >>> validate_date_format("2025/08/13")
        True
        >>> validate_date_format("2025/13/01")  # Invalid month
        False
        >>> validate_date_format("not-a-date")
        False
    """
    try:
        normalize_date_format(date_str)
        return True
    except ValueError:
        return False


# Convenience functions for common operations
def to_aws_format(date_str: str) -> str:
    """Convert any date format to AWS format (YYYY/MM/DD)."""
    aws_format, _, _ = normalize_date_format(date_str)
    return aws_format


def to_local_format(date_str: str) -> str:
    """Convert any date format to local format (YYYY-MM-DD)."""
    _, local_format, _ = normalize_date_format(date_str)
    return local_format


def to_compact_format(date_str: str) -> str:
    """Convert any date format to compact format (YYYYMMDD)."""
    _, _, compact_format = normalize_date_format(date_str)
    return compact_format


if __name__ == "__main__":
    # Test the functions
    test_dates = ["2025/08/13", "2025-08-13", "20250813"]
    
    print("🧪 Testing normalize_date_format():")
    for test_date in test_dates:
        try:
            aws, local, compact = normalize_date_format(test_date)
            print(f"  Input: {test_date:12} → AWS: {aws}, Local: {local}, Compact: {compact}")
        except ValueError as e:
            print(f"  Input: {test_date:12} → ERROR: {e}")
    
    print(f"\n📅 Yesterday: {get_yesterday_formats()}")
    print(f"📅 Today: {get_today_formats()}")
    
    print("\n✅ Date utility functions ready!")
