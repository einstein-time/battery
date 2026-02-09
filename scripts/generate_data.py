#!/usr/bin/env python3
"""CLI script for generating thermal simulation training data.

Generates trajectories of 2D heat equation solutions with randomized
parameters and saves to compressed HDF5 format.

Usage:
    python scripts/generate_data.py --config configs/generate.yaml
    python scripts/generate_data.py --quick_test
    python scripts/generate_data.py --n_trajectories 100 --output data/small.h5
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, DEFAULTS
from src.utils.logging import setup_logging
from src.data_utils.generator import DataGenerator

logger = logging.getLogger("battery_surrogate")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate thermal simulation training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generate.yaml",
        help="Path to generation config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output HDF5 file path",
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=None,
        help="Override number of trajectories to generate",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Generate small test dataset (10 trajectories, 50 steps, 32x32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for data generation."""
    args = parse_args()

    setup_logging(level=logging.INFO)

    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        logger.warning("Config file not found: %s, using defaults", config_path)
        config = DEFAULTS.copy()

    if args.quick_test:
        config["data"]["n_trajectories"] = 10
        config["data"]["n_steps"] = 50
        config["data"]["grid_size"] = 32
        output_path = Path("data/raw/quick_test.h5")
        logger.info("Quick test mode: 10 trajectories, 50 steps, 32x32 grid")
    else:
        output_path = Path(config["data"].get("output_path", "data/raw/thermal_dataset.h5"))

    if args.output:
        output_path = Path(args.output)
    if args.n_trajectories:
        config["data"]["n_trajectories"] = args.n_trajectories
    if args.n_workers:
        config["data"]["n_workers"] = args.n_workers

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Battery Thermal Data Generation")
    logger.info("=" * 60)
    logger.info("Trajectories:  %d", config["data"]["n_trajectories"])
    logger.info("Grid size:     %d", config["data"].get("grid_size", 128))
    logger.info("Output:        %s", output_path)
    logger.info("=" * 60)

    generator = DataGenerator(config)

    start_time = time.time()

    if args.quick_test:
        generator.generate_quick_test(output_path)
    else:
        n_workers = config["data"].get("n_workers", 4)
        generator.generate_dataset(
            n_trajectories=config["data"]["n_trajectories"],
            output_path=output_path,
            n_workers=n_workers,
        )

    elapsed = time.time() - start_time
    logger.info("Data generation complete in %.1f seconds", elapsed)
    logger.info("Saved to: %s", output_path)


if __name__ == "__main__":
    main()
