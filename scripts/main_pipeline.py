#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
Runs the complete S2 data processing pipeline (Steps 3-6).

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
            'local_format': yesterday.strftime('%Y-%m-%d')  # For Steps 4-6
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
    
    def run_pipeline(self, date_aws_format: str = None, date_local_format: str = None, 
                    skip_steps: list = None, only_steps: list = None) -> bool:
        """
        Run the complete pipeline.
        
        Args:
            date_aws_format: Date in YYYY/MM/DD format for AWS operations
            date_local_format: Date in YYYY-MM-DD format for local operations
            skip_steps: List of step numbers to skip (e.g., [3, 4])
            only_steps: List of step numbers to run only (e.g., [5, 6])
        """
        # Get yesterday's date if not provided
        if not date_aws_format or not date_local_format:
            yesterday = self.get_yesterday()
            date_aws_format = date_aws_format or yesterday['aws_format']
            date_local_format = date_local_format or yesterday['local_format']
        
        skip_steps = skip_steps or []
        
        # Determine which steps to run
        all_steps = [3, 4, 5, 6]
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
        
        return pipeline_success
    
    def _should_stop_on_failure(self) -> bool:
        """Determine if pipeline should stop on step failure."""
        # For automated runs, continue to next step even if one fails
        # This allows partial processing
        return False
    
    def run_interactive(self):
        """Run pipeline in interactive mode."""
        try:
            print("🚀 S2 Data Processing Pipeline")
            print("=" * 50)
            
            # Get date options
            yesterday = self.get_yesterday()
            
            print(f"\nDate options:")
            print(f"1. Yesterday: {yesterday['local_format']}")
            print(f"2. Custom date")
            
            date_choice = input("Select date option (1-2) [default: 1]: ").strip()
            
            if date_choice == '2':
                date_input = input("Enter date (YYYY-MM-DD): ").strip()
                try:
                    date_obj = datetime.strptime(date_input, '%Y-%m-%d')
                    date_local_format = date_input
                    date_aws_format = date_obj.strftime('%Y/%m/%d')
                except ValueError:
                    print("Invalid date format. Using yesterday.")
                    date_local_format = yesterday['local_format']
                    date_aws_format = yesterday['aws_format']
            else:
                date_local_format = yesterday['local_format']
                date_aws_format = yesterday['aws_format']
            
            # Get step options
            print(f"\nStep options:")
            print("1. Run all steps (3-6)")
            print("2. Run specific steps")
            print("3. Skip specific steps")
            
            step_choice = input("Select step option (1-3) [default: 1]: ").strip()
            
            skip_steps = None
            only_steps = None
            
            if step_choice == '2':
                steps_input = input("Enter steps to run (e.g., 3,4,5,6): ").strip()
                try:
                    only_steps = [int(s.strip()) for s in steps_input.split(',')]
                except ValueError:
                    print("Invalid input. Running all steps.")
            elif step_choice == '3':
                steps_input = input("Enter steps to skip (e.g., 3,4): ").strip()
                try:
                    skip_steps = [int(s.strip()) for s in steps_input.split(',')]
                except ValueError:
                    print("Invalid input. Running all steps.")
            
            # Run pipeline with logging context
            with PipelineLogger('S2_Pipeline', self.config.get_log_config()) as logger:
                self.logger = logger
                success = self.run_pipeline(
                    date_aws_format=date_aws_format,
                    date_local_format=date_local_format,
                    skip_steps=skip_steps,
                    only_steps=only_steps
                )
                
                if success:
                    print("\n🎉 Pipeline completed successfully!")
                else:
                    print("\n⚠️  Pipeline completed with some failures. Check logs for details.")
            
        except KeyboardInterrupt:
            print("\n❌ Pipeline interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            sys.exit(1)
    
    def run_automated(self, date_local_format: str = None, skip_steps: list = None) -> bool:
        """Run pipeline in automated mode (for cron scheduling)."""
        try:
            # Get yesterday's date if not provided
            if not date_local_format:
                yesterday = self.get_yesterday()
                date_local_format = yesterday['local_format']
                date_aws_format = yesterday['aws_format']
            else:
                # Convert local format to AWS format
                date_obj = datetime.strptime(date_local_format, '%Y-%m-%d')
                date_aws_format = date_obj.strftime('%Y/%m/%d')
            
            # Run pipeline with logging context
            with PipelineLogger('S2_Pipeline_Automated', self.config.get_log_config()) as logger:
                self.logger = logger
                success = self.run_pipeline(
                    date_aws_format=date_aws_format,
                    date_local_format=date_local_format,
                    skip_steps=skip_steps
                )
                
                return success
                
        except Exception as e:
            print(f"❌ Automated pipeline error: {e}")
            return False


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='S2 Data Processing Pipeline')
    parser.add_argument('--date', type=str, help='Date to process (YYYY-MM-DD). Defaults to yesterday.')
    parser.add_argument('--automated', action='store_true', help='Run in automated mode (no user prompts)')
    parser.add_argument('--skip-steps', type=str, help='Comma-separated list of steps to skip (e.g., 3,4)')
    parser.add_argument('--only-steps', type=str, help='Comma-separated list of steps to run only (e.g., 5,6)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        pipeline = S2DataPipeline(args.config)
        
        # Parse step arguments
        skip_steps = None
        only_steps = None
        
        if args.skip_steps:
            try:
                skip_steps = [int(s.strip()) for s in args.skip_steps.split(',')]
            except ValueError:
                print("❌ Invalid skip-steps format. Use comma-separated numbers (e.g., 3,4)")
                sys.exit(1)
        
        if args.only_steps:
            try:
                only_steps = [int(s.strip()) for s in args.only_steps.split(',')]
            except ValueError:
                print("❌ Invalid only-steps format. Use comma-separated numbers (e.g., 5,6)")
                sys.exit(1)
        
        if args.automated:
            success = pipeline.run_automated(args.date, skip_steps)
            sys.exit(0 if success else 1)
        else:
            if args.date or skip_steps or only_steps:
                # Run with specific parameters
                yesterday = pipeline.get_yesterday()
                date_local_format = args.date or yesterday['local_format']
                
                try:
                    date_obj = datetime.strptime(date_local_format, '%Y-%m-%d')
                    date_aws_format = date_obj.strftime('%Y/%m/%d')
                except ValueError:
                    print("❌ Invalid date format. Use YYYY-MM-DD")
                    sys.exit(1)
                
                with PipelineLogger('S2_Pipeline', pipeline.config.get_log_config()) as logger:
                    pipeline.logger = logger
                    success = pipeline.run_pipeline(
                        date_aws_format=date_aws_format,
                        date_local_format=date_local_format,
                        skip_steps=skip_steps,
                        only_steps=only_steps
                    )
                    
                    if success:
                        print("🎉 Pipeline completed successfully!")
                    else:
                        print("⚠️  Pipeline completed with failures. Check logs for details.")
                        sys.exit(1)
            else:
                # Run interactive mode
                pipeline.run_interactive()
                
    except Exception as e:
        print(f"❌ Fatal pipeline error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()