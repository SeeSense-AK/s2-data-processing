#!/usr/bin/env python3
"""
Main Pipeline Orchestrator (Updated with Step 7)
Runs the complete S2 data processing pipeline (Steps 3-7).

Author: SeeSense Data Pipeline
"""

import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_manager import ConfigManager
from scripts.utils.logger_setup import PipelineLogger


class S2DataPipeline:
    """Main pipeline orchestrator for S2 data processing."""
    
    def __init__(self, config_path=None):
        """Initialize the pipeline with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = None  # Will be set in context manager
        
    def get_yesterday(self):
        """Get yesterday's date in YYYY/MM/DD and YYYY-MM-DD formats."""
        yesterday = datetime.utcnow() - timedelta(days=1)
        return {
            'aws_format': yesterday.strftime('%Y/%m/%d'),  # For Step 3
            'local_format': yesterday.strftime('%Y-%m-%d')  # For Steps 4-7
        }
    
    def run_step3_daily_combiner(self, date_aws_format: str) -> bool:
        """Run Step 3: Daily CSV Combiner."""
        try:
            self.logger.info("🔄 Starting Step 3: Daily CSV Combiner")
            
            from scripts.step3_daily_combiner import DailyCombiner
            
            combiner = DailyCombiner()
            success = combiner.run_automated(date_aws_format)
            
            if success:
                self.logger.info("✅ Step 3 completed successfully")
                return True
            else:
                self.logger.error("❌ Step 3 failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Step 3 error: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_step4_device_bifurcation(self, date_local_format: str) -> bool:
        """Run Step 4: Device Bifurcation."""
        try:
            self.logger.info("🔄 Starting Step 4: Device Bifurcation")
            
            from scripts.step4_device_bifurcation import DeviceBifurcator
            
            bifurcator = DeviceBifurcator()
            success = bifurcator.run_automated(date_local_format)
            
            if success:
                self.logger.info("✅ Step 4 completed successfully")
                return True
            else:
                self.logger.error("❌ Step 4 failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Step 4 error: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_step5_interpolation(self, date_local_format: str) -> bool:
        """Run Step 5: OSRM Interpolation."""
        try:
            self.logger.info("🔄 Starting Step 5: OSRM Interpolation")
            
            from scripts.step5_interpolation import OSRMInterpolator
            
            interpolator = OSRMInterpolator()
            success = interpolator.run_automated(date_local_format)
            
            if success:
                self.logger.info("✅ Step 5 completed successfully")
                return True
            else:
                self.logger.error("❌ Step 5 failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Step 5 error: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_step6_combine_upload(self, date_local_format: str) -> bool:
        """Run Step 6: Combine and Upload."""
        try:
            self.logger.info("🔄 Starting Step 6: Combine and Upload")
            
            from scripts.step6_combine_upload import RegionalCombiner
            
            combiner = RegionalCombiner()
            success = combiner.run_automated(date_local_format)
            
            if success:
                self.logger.info("✅ Step 6 completed successfully")
                return True
            else:
                self.logger.error("❌ Step 6 failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Step 6 error: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_step7_abnormal_events(self, date_local_format: str) -> bool:
        """Run Step 7: Abnormal Events Detection."""
        try:
            self.logger.info("🔄 Starting Step 7: Abnormal Events Detection")
            
            from scripts.step7_abnormal_events import AbnormalEventsDetector
            
            detector = AbnormalEventsDetector()
            success = detector.run_automated(date_local_format)
            
            if success:
                self.logger.info("✅ Step 7 completed successfully")
                return True
            else:
                self.logger.warning("⚠️ Step 7 failed (but pipeline continues)")
                return True  # Don't fail the pipeline if abnormal events fails
                
        except Exception as e:
            self.logger.error(f"❌ Step 7 error: {e}")
            self.logger.debug(traceback.format_exc())
            self.logger.warning("⚠️ Step 7 failed (but pipeline continues)")
            return True  # Don't fail the pipeline if abnormal events fails
    
    def run_pipeline(self, date_aws_format: str = None, date_local_format: str = None, 
                    skip_steps: list = None, only_steps: list = None) -> bool:
        """
        Run the complete pipeline.
        
        Args:
            date_aws_format: Date in YYYY/MM/DD format for AWS operations
            date_local_format: Date in YYYY-MM-DD format for local operations
            skip_steps: List of step numbers to skip (e.g., [3, 4])
            only_steps: List of step numbers to run only (e.g., [5, 6, 7])
        """
        # Get yesterday's date if not provided
        if not date_aws_format or not date_local_format:
            yesterday = self.get_yesterday()
            date_aws_format = date_aws_format or yesterday['aws_format']
            date_local_format = date_local_format or yesterday['local_format']
        
        skip_steps = skip_steps or []
        
        # Determine which steps to run
        all_steps = [3, 4, 5, 6, 7]  # Added Step 7
        if only_steps:
            steps_to_run = [step for step in only_steps if step in all_steps]
        else:
            steps_to_run = [step for step in all_steps if step not in skip_steps]
        
        self.logger.info(f"📅 Processing date: {date_local_format} (AWS: {date_aws_format})")
        self.logger.info(f"🎯 Steps to run: {steps_to_run}")
        
        # Pipeline execution
        pipeline_success = True
        
        # Step 3: Daily CSV Combiner
        if 3 in steps_to_run:
            if not self.run_step3_daily_combiner(date_aws_format):
                pipeline_success = False
                if self._should_stop_on_failure():
                    return False
        
        # Step 4: Device Bifurcation
        if 4 in steps_to_run:
            if not self.run_step4_device_bifurcation(date_local_format):
                pipeline_success = False
                if self._should_stop_on_failure():
                    return False
        
        # Step 5: OSRM Interpolation
        if 5 in steps_to_run:
            if not self.run_step5_interpolation(date_local_format):
                pipeline_success = False
                if self._should_stop_on_failure():
                    return False
        
        # Step 6: Combine and Upload
        if 6 in steps_to_run:
            if not self.run_step6_combine_upload(date_local_format):
                pipeline_success = False
                if self._should_stop_on_failure():
                    return False
        
        # Step 7: Abnormal Events Detection (New)
        if 7 in steps_to_run:
            # Note: Step 7 doesn't fail the pipeline even if it encounters errors
            # This ensures that missing accelerometer data doesn't break the existing workflow
            step7_success = self.run_step7_abnormal_events(date_local_format)
            if not step7_success:
                self.logger.warning("⚠️ Step 7 had issues but pipeline continues")
                # Don't set pipeline_success to False for Step 7
        
        return pipeline_success
    
    def _should_stop_on_failure(self) -> bool:
        """Determine if pipeline should stop on step failure."""
        # For automated runs, continue to next step even if one fails
        # This allows partial processing
        return False
    
    def run_interactive(self):
        """Run pipeline in interactive mode."""
        try:
            print("🚀 S2 Data Processing Pipeline (Steps 3-7)")
            print("=" * 50)
            
            # Get date options
            yesterday = self.get_yesterday()
            
            print(f"\nDate options:")
            print(f"1. Yesterday: {yesterday['local_format']}")
            print(f"2. Today: {datetime.utcnow().strftime('%Y-%m-%d')}")
            print(f"3. Custom date")
            
            choice = input(f"Select option (1-3) or press Enter for yesterday [{yesterday['local_format']}]: ").strip()
            
            if choice == '1' or choice == '':
                date_aws = yesterday['aws_format']
                date_local = yesterday['local_format']
            elif choice == '2':
                today = datetime.utcnow()
                date_aws = today.strftime('%Y/%m/%d')
                date_local = today.strftime('%Y-%m-%d')
            elif choice == '3':
                date_input = input("Enter custom date (YYYY-MM-DD): ").strip()
                if date_input:
                    try:
                        date_obj = datetime.strptime(date_input, '%Y-%m-%d')
                        date_aws = date_obj.strftime('%Y/%m/%d')
                        date_local = date_input
                    except ValueError:
                        print("Invalid date format. Using yesterday.")
                        date_aws = yesterday['aws_format']
                        date_local = yesterday['local_format']
                else:
                    date_aws = yesterday['aws_format']
                    date_local = yesterday['local_format']
            else:
                date_aws = yesterday['aws_format']
                date_local = yesterday['local_format']
            
            # Step selection
            print(f"\nStep options:")
            print(f"1. Run all steps (3-7)")
            print(f"2. Run core pipeline only (3-6)")
            print(f"3. Run abnormal events only (7)")
            print(f"4. Custom step selection")
            
            step_choice = input("Select option (1-4) or press Enter for all steps [1]: ").strip()
            
            skip_steps = []
            only_steps = None
            
            if step_choice == '2':
                skip_steps = [7]
            elif step_choice == '3':
                only_steps = [7]
            elif step_choice == '4':
                steps_input = input("Enter steps to run (e.g., 3,4,5,6,7): ").strip()
                if steps_input:
                    try:
                        only_steps = [int(x.strip()) for x in steps_input.split(',')]
                        only_steps = [x for x in only_steps if x in [3, 4, 5, 6, 7]]
                    except ValueError:
                        print("Invalid input. Running all steps.")
            
            # Run pipeline with PipelineLogger context manager
            with PipelineLogger(self.config.get_log_config()) as logger:
                self.logger = logger
                success = self.run_pipeline(date_aws, date_local, skip_steps, only_steps)
            
            if success:
                print("✅ Pipeline completed successfully!")
            else:
                print("❌ Pipeline completed with some failures!")
            
            return success
            
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info("❌ Pipeline interrupted by user")
            return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Unexpected error in interactive mode: {e}")
                self.logger.debug(traceback.format_exc())
            return False
    
    def run_automated(self, date_aws_format: str = None, date_local_format: str = None, 
                     skip_steps: list = None, only_steps: list = None) -> bool:
        """Run pipeline in automated mode (for cron jobs)."""
        with PipelineLogger(self.config.get_log_config()) as logger:
            self.logger = logger
            return self.run_pipeline(date_aws_format, date_local_format, skip_steps, only_steps)


def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='S2 Data Processing Pipeline (Steps 3-7)')
    parser.add_argument('--date', type=str, help='Date to process (YYYY-MM-DD format)')
    parser.add_argument('--automated', action='store_true', help='Run in automated mode')
    parser.add_argument('--skip-steps', type=str, help='Comma-separated list of steps to skip (e.g., 3,4)')
    parser.add_argument('--only-steps', type=str, help='Comma-separated list of steps to run only (e.g., 5,6,7)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Parse step arguments
    skip_steps = []
    only_steps = None
    
    if args.skip_steps:
        try:
            skip_steps = [int(x.strip()) for x in args.skip_steps.split(',')]
        except ValueError:
            print("Error: Invalid skip-steps format. Use comma-separated numbers (e.g., 3,4)")
            sys.exit(1)
    
    if args.only_steps:
        try:
            only_steps = [int(x.strip()) for x in args.only_steps.split(',')]
        except ValueError:
            print("Error: Invalid only-steps format. Use comma-separated numbers (e.g., 5,6,7)")
            sys.exit(1)
    
    # Initialize pipeline
    pipeline = S2DataPipeline(args.config)
    
    if args.automated:
        # Automated mode - use yesterday if no date specified
        if args.date:
            date_obj = datetime.strptime(args.date, '%Y-%m-%d')
            date_aws = date_obj.strftime('%Y/%m/%d')
            date_local = args.date
        else:
            yesterday = pipeline.get_yesterday()
            date_aws = yesterday['aws_format']
            date_local = yesterday['local_format']
        
        success = pipeline.run_automated(date_aws, date_local, skip_steps, only_steps)
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        if args.date:
            try:
                date_obj = datetime.strptime(args.date, '%Y-%m-%d')
                date_aws = date_obj.strftime('%Y/%m/%d')
                date_local = args.date
                
                with PipelineLogger(pipeline.config.get_log_config()) as logger:
                    pipeline.logger = logger
                    success = pipeline.run_pipeline(date_aws, date_local, skip_steps, only_steps)
                
                print("✅ Pipeline completed successfully!" if success else "❌ Pipeline completed with failures!")
                sys.exit(0 if success else 1)
            except ValueError:
                print("Error: Invalid date format. Use YYYY-MM-DD")
                sys.exit(1)
        else:
            success = pipeline.run_interactive()
            sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
