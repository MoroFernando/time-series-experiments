"""
Entry point for the dimensionality reduction classification experiments.

Usage
-----
    python run_experiment.py                    # uses config.yaml
    python run_experiment.py --config my.yaml   # custom config
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel",       "3")

import argparse
import datetime
import sys

import torch
import yaml

from src.classifiers import get_classifiers
from src.experiment import run_experiment
from src import reduction


class _TeeLogger:
    """Mirrors a stream to both the terminal and a log file.

    Progress-bar lines (written with \\r) are collapsed: only the final
    state of each overwritten line is saved to the log file, keeping it
    human-readable without thousands of intermediate bar frames.
    """

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file
        self._buf = ""

    def _flush_line(self, line: str) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.write(f"[{ts}] {line}\n")
        self._log.flush()

    def write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()
        for ch in text:
            if ch == "\r":
                self._buf = ""
            elif ch == "\n":
                self._flush_line(self._buf)
                self._buf = ""
            else:
                self._buf += ch

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except AttributeError:
            return False

ALL_REDUCTION_METHODS = {
    "PAA":    reduction.PAA_reduce,
    "DFT":    reduction.DFT_reduce,
    "DWT":    reduction.DWT_reduce,
    "SVD":    reduction.SVD_reduce,
    "PCA":    reduction.PCA_reduce,
    "KPCA":   reduction.KPCA_reduce,
    "Isomap": reduction.Isomap_reduce,
    "AE":     reduction.AEReducer,
    "AE-SIT": reduction.AE_SIT_reduce,
    "CAE":     reduction.CAEReducer,
    "CAE-SIT": reduction.CAE_SIT_reduce,
    "TCN":     reduction.TCNReducer,
    "TCN-SIT": reduction.TCN_SIT_reduce,
    "S2V":    reduction.Series2VecReducer,
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

    out_dir = os.path.dirname(output_file) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Mirror all output (stdout + stderr) to experiment.log.
    # Progress-bar lines (\\r) are collapsed so the file stays readable.
    log_path = os.path.join(out_dir, "experiment.log")
    _log_file = open(log_path, "a", encoding="utf-8")
    _log_file.write(f"\n{'='*60}\n")
    _log_file.write(f"Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_file.write(f"Config: {args.config}\n")
    _log_file.write(f"{'='*60}\n")
    _log_file.flush()
    sys.stdout = _TeeLogger(sys.__stdout__, _log_file)
    sys.stderr = _TeeLogger(sys.__stderr__, _log_file)

    print("=" * 60)
    print("Dimensionality Reduction Experiments")
    print(f"  Config          : {args.config}")
    print(f"  Log             : {log_path}")
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

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_file.write(f"Run finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_file.close()

if __name__ == "__main__":
    main()
