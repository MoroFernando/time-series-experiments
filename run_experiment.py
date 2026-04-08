"""
Entry point for the dimensionality reduction classification experiments.

Usage
-----
    python run_experiment.py                    # uses config.yaml
    python run_experiment.py --config my.yaml   # custom config
"""

import argparse
import os

import yaml

from src.classifiers import get_classifiers
from src.experiment import run_experiment
from src.reduction import (
    AE_reduce,
    CAE_reduce,
    DFT_reduce,
    DWT_reduce,
    Isomap_reduce,
    KPCA_reduce,
    PAA_reduce,
    PCA_reduce,
    SVD_reduce,
)

ALL_REDUCTION_METHODS = {
    "PAA": PAA_reduce,
    "DFT": DFT_reduce,
    "DWT": DWT_reduce,
    "SVD": SVD_reduce,
    "PCA": PCA_reduce,
    "KPCA": KPCA_reduce,
    "Isomap": Isomap_reduce,
    "AE": AE_reduce,
    "CAE": CAE_reduce,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dimensionality reduction + classification experiments."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    datasets = cfg["datasets"]
    retention_rates = cfg["retention_rates"]
    random_state = cfg.get("reproducibility", {}).get("random_state", 1)
    output_file = cfg["output"]["results_file"]

    selected_methods = cfg["reduction_methods"]
    unknown_methods = set(selected_methods) - set(ALL_REDUCTION_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown reduction methods in config: {unknown_methods}")
    reduction_methods = {name: ALL_REDUCTION_METHODS[name] for name in selected_methods}

    selected_classifiers = cfg["classifiers"]
    unknown_clfs = set(selected_classifiers) - set(get_classifiers().keys())
    if unknown_clfs:
        raise ValueError(f"Unknown classifiers in config: {unknown_clfs}")

    def classifiers_factory():
        return {k: v for k, v in get_classifiers(random_state).items() if k in selected_classifiers}

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    print("=" * 60)
    print("Dimensionality Reduction Experiments")
    print(f"  Config          : {args.config}")
    print(f"  Datasets        : {datasets}")
    print(f"  Classifiers     : {selected_classifiers}")
    print(f"  Methods         : {list(reduction_methods)}")
    print(f"  Retention rates : {retention_rates}")
    print(f"  Output          : {output_file}")
    print("=" * 60)

    df = run_experiment(
        datasets=datasets,
        classifiers_factory=classifiers_factory,
        reduction_methods=reduction_methods,
        retention_rates=retention_rates,
        output_file=output_file,
    )

    print("\n--- Results summary ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
