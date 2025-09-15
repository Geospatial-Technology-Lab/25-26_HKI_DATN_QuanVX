#!/usr/bin/env python3
"""
Script chính để chạy optimization experiments
"""

import argparse
import sys
from pathlib import Path

# Thêm đường dẫn ml vào Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.yaml_config import load_optimization_config, get_optimization_params
from utils.logging import setup_logging, get_logger

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run ML Optimization Experiments")
    
    parser.add_argument(
        "--model", 
        choices=["rf", "svm", "xgb", "mlp"],
        required=True,
        help="Model to optimize"
    )
    
    parser.add_argument(
        "--optimizer",
        choices=["pso", "rso", "puma"], 
        required=True,
        help="Optimization algorithm"
    )
    
    parser.add_argument(
        "--config",
        help="Path to custom config file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=getattr(logging, log_level))
    logger = get_logger(__name__)
    
    logger.info(f"Starting optimization: {args.model} with {args.optimizer}")
    
    # Load config
    try:
        opt_config = get_optimization_params(args.optimizer)
        logger.info(f"Loaded config for {args.optimizer}: {opt_config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # Import và chạy experiment tương ứng
    try:
        if args.optimizer == "pso":
            if args.model == "rf":
                from experiments.pso_rf import main as run_experiment
            elif args.model == "svm":
                from experiments.pso_svm import main as run_experiment
            elif args.model == "xgb":
                from experiments.pso_xgb import main as run_experiment
            elif args.model == "mlp":
                from experiments.pso_mlp import main as run_experiment
                
        elif args.optimizer == "rso":
            if args.model == "rf":
                from experiments.rso_rf import main as run_experiment
            elif args.model == "svm":
                from experiments.rso_svm import main as run_experiment
            elif args.model == "xgb":
                from experiments.rso_xgb import main as run_experiment
            elif args.model == "mlp":
                from experiments.rso_mlp import main as run_experiment
                
        elif args.optimizer == "puma":
            if args.model == "rf":
                from experiments.po_rf import main as run_experiment
            elif args.model == "svm":
                from experiments.po_svm import main as run_experiment
            elif args.model == "xgb":
                from experiments.po_xgb import main as run_experiment
            elif args.model == "mlp":
                from experiments.po_mlp import main as run_experiment
        
        # Chạy experiment
        result = run_experiment()
        logger.info(f"Experiment completed successfully")
        logger.info(f"Best fitness: {result.get('best_fitness', 'N/A')}")
        
        return 0
        
    except ImportError as e:
        logger.error(f"Cannot import experiment module: {e}")
        return 1
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return 1

if __name__ == "__main__":
    import logging
    exit(main())