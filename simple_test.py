#!/usr/bin/env python3
import os
import sys
import datetime

# Simple test script
log_file = "/tmp/pipeline_simple_test.log"

with open(log_file, "a") as f:
    f.write(f"{datetime.datetime.now()}: Script started - Python {sys.version}\n")
    f.write(f"Working dir: {os.getcwd()}\n")
    f.write(f"User: {os.getenv('USER', 'unknown')}\n")
    f.write(f"PATH: {os.getenv('PATH', '')}\n")
    
    # Test basic imports that your main script might need
    try:
        import pandas
        f.write("Pandas import: SUCCESS\n")
    except ImportError as e:
        f.write(f"Pandas import: FAILED - {e}\n")
    
    f.write(f"Script completed successfully\n")

print("Simple test completed - check /tmp/pipeline_simple_test.log")
