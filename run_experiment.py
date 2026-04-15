"""
Entry point for the dimensionality reduction classification experiments.

Usage
-----
    python run_experiment.py                    # uses config.yaml
    python run_experiment.py --config my.yaml   # custom config
"""

# ── Silence TensorFlow / oneDNN C++ logging ───────────────────────────────
# These variables MUST be set before any import that loads the TF C++ runtime
# (including indirect imports via aeon.classification.deep_learning).
#
#   TF_CPP_MIN_LOG_LEVEL  0=DEBUG 1=INFO 2=WARNING 3=ERROR (only FATAL shown)
#   TF_ENABLE_ONEDNN_OPTS 0  →  disables oneDNN and suppresses its banner
#   GLOG_minloglevel      3  →  silences absl/glog C++ messages (incl. the
#                               "before absl::InitializeLog" bootstrap warning)
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel",       "3")
# ──────────────────────────────────────────────────────────────────────────

import argparse

import torch
import yaml

from src.classifiers import get_classifiers
from src.experiment import run_experiment
from src.reduction import (
    PAA_reduce,
    DFT_reduce,
    DWT_reduce,
    SVD_reduce,
    PCA_reduce,
    KPCA_reduce,
    Isomap_reduce,
    AEReducer,
    AE_SIT_reduce,
    CAEReducer,
    CAE_SIT_reduce,
    TCNReducer,
    TCN_SIT_reduce,
)

ALL_REDUCTION_METHODS = {
    "PAA":    PAA_reduce,
    "DFT":    DFT_reduce,
    "DWT":    DWT_reduce,
    "SVD":    SVD_reduce,
    "PCA":    PCA_reduce,
    "KPCA":   KPCA_reduce,
    "Isomap": Isomap_reduce,
    "AE":     AEReducer,
    "AE-SIT": AE_SIT_reduce,
    "CAE":     CAEReducer,
    "CAE-SIT": CAE_SIT_reduce,
    "TCN":     TCNReducer,
    "TCN-SIT": TCN_SIT_reduce,
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


def print_gpu_info() -> None:
    """Print detected GPU(s) for PyTorch and TensorFlow."""

    # PyTorch
    if torch.cuda.is_available():
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"  PyTorch GPU(s)  : {', '.join(gpus)}")
    else:
        print("  PyTorch GPU     : not detected (CPU only)")

    # TensorFlow — imported locally to avoid paying the TF startup cost upfront
    try:
        import tensorflow as tf
        tf_gpus = tf.config.list_physical_devices("GPU")
        if tf_gpus:
            names = [g.name for g in tf_gpus]
            print(f"  TensorFlow GPU  : {', '.join(names)}")
        else:
            print("  TensorFlow GPU  : not detected (CPU only)")
    except Exception:
        print("  TensorFlow GPU  : could not query (TF not installed?)")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    datasets = cfg["datasets"]
    retention_rates = cfg["retention_rates"]
    random_state = cfg.get("reproducibility", {}).get("random_state", 1)
    output_file = cfg["output"]["results_file"]
    neighborhood_file = cfg["output"]["neighborhood_file"]
    neighborhood_ks = cfg["output"].get("neighborhood_ks", [5])

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
    print(f"  Neighborhood    : {neighborhood_file} (ks={neighborhood_ks})")
    print_gpu_info()
    print("=" * 60)

    df = run_experiment(
        datasets=datasets,
        classifiers_factory=classifiers_factory,
        reduction_methods=reduction_methods,
        retention_rates=retention_rates,
        output_file=output_file,
        neighborhood_file=neighborhood_file,
        neighborhood_ks=neighborhood_ks,
    )

if __name__ == "__main__":
    main()
