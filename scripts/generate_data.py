#!/usr/bin/env python
"""Generate synthetic thermal simulation dataset."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils.generator import DataGenerator
from src.utils.config import load_config, merge_configs
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Generate thermal simulation dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generate.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path from config",
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=None,
        help="Override number of trajectories from config",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Use quick test configuration (10 trajectories, 32x32 grid)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, verbose=args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("THERMAL DATA GENERATION")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(args.config)

    # Apply quick test overrides
    if args.quick_test:
        logger.info("Using quick test configuration")
        quick_config = config.get("quick_test", {})
        config["data"]["n_trajectories"] = quick_config.get("n_trajectories", 10)
        config["data"]["n_steps"] = quick_config.get("n_steps", 50)
        config["data"]["grid_size"] = quick_config.get("grid_size", 32)

    # Apply command-line overrides
    if args.output:
        config["data"]["output_path"] = str(args.output)

    if args.n_trajectories:
        config["data"]["n_trajectories"] = args.n_trajectories

    # Create generator
    generator = DataGenerator(config)

    # Generate dataset
    try:
        generator.generate_dataset()
        logger.info("Data generation completed successfully!")
    except Exception as e:
        logger.error(f"Data generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
