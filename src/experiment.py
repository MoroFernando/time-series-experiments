"""
Core experiment logic: dimensionality reduction, classifier training, and result persistence.
"""
import gc
import os
import sys
import time
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------

def reduce_dataset(
    method_name: str,
    method,
    X_train: np.ndarray,
    X_test: np.ndarray,
    retention_rate: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Apply a reduction method to all series in train and test sets.

    Parameters
    ----------
    method_name : human-readable label used in logging
    method : callable(series, w) -> reduced_series
    X_train / X_test : arrays of shape (n_samples, n_channels, n_timepoints)
    retention_rate : fraction of timepoints to keep (e.g. 0.5 = halve the series)

    Returns
    -------
    X_train_reduced, X_test_reduced : reduced arrays
    duration_s : wall-clock seconds spent on reduction (train + test)
    """
    series_len = X_train.shape[2]
    w = max(1, round(series_len * retention_rate))
    print(f"\n[reduction] {method_name} | retention={retention_rate} | {series_len} -> {w} timepoints")

    import inspect

    # If the method is a class (global-style reducer), instantiate it fresh so
    # each (dataset, retention_rate) combination gets its own model.
    if inspect.isclass(method):
        method = method()

    start = time.time()
    if hasattr(method, "fit_transform"):
        # Global method: train once on all training series, then encode both splits.
        X_train_red = method.fit_transform(X_train, w)
        print(f"  Train: done ({len(X_train)} samples)")
        X_test_red = method.transform(X_test, w)
        print(f"  Test : done ({len(X_test)} samples)")
    else:
        X_train_red = _apply_reduction(method, X_train, w, label="Train")
        X_test_red = _apply_reduction(method, X_test, w, label="Test ")
    duration = round(time.time() - start, 2)

    print(f"[reduction] Done in {duration}s")
    return X_train_red, X_test_red, duration


def _apply_reduction(method, data: np.ndarray, w: int, label: str) -> np.ndarray:
    """Iterate over samples/channels with a progress bar, applying `method`."""
    total = len(data)
    reduced = []

    for i, sample in enumerate(data):
        reduced_sample = []
        for j, series in enumerate(sample):
            reduced_sample.append(method(series, w))

            # Free GPU memory periodically when using neural methods
            if torch.cuda.is_available() and (i * len(sample) + j) % 50 == 0:
                torch.cuda.empty_cache()

        reduced.append(reduced_sample)
        _print_progress(label, i + 1, total)

    print()  # newline after progress bar
    return np.array(reduced)


def _print_progress(label: str, done: int, total: int, bar_len: int = 30) -> None:
    pct = done / total
    filled = int(pct * bar_len)
    bar = "█" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r  {label}: [{bar}] {pct:>5.1%} ({done}/{total})")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _lite_worker(clf, X_train, y_train, X_test, y_test, queue):
    """
    Worker function executed in a separate process for LITE (TensorFlow).

    TensorFlow does not release GPU/RAM memory after training even with explicit
    deletion. Running it in a subprocess guarantees full memory release when the
    process exits.
    """
    try:
        import os
        import warnings
        import time
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("GLOG_minloglevel",       "3")
        import tensorflow as tf
        from sklearn.metrics import accuracy_score

        # Allow TF to allocate GPU memory incrementally instead of all at once
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="unsafe cast", module="numba")
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_time = round(time.time() - t0, 2)

            t0 = time.time()
            y_pred = clf.predict(X_test)
            test_time = round(time.time() - t0, 2)

        acc = accuracy_score(y_test, y_pred)
        queue.put((acc, train_time, test_time))
    except Exception as e:
        queue.put(e)


def train_and_evaluate(
    clf_name: str,
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    """
    Fit a classifier and evaluate it.

    LITE is run in an isolated subprocess so TensorFlow releases all GPU/RAM
    memory when the process exits. All other classifiers run in-process.

    Returns
    -------
    accuracy : float
    train_time_s : wall-clock seconds for clf.fit()
    test_time_s : wall-clock seconds for clf.predict()
    """
    print(f"  [clf] Training {clf_name}...")

    if clf_name == "LITE":
        acc, train_time, test_time = _run_lite_in_subprocess(clf, X_train, y_train, X_test, y_test)
    else:
        acc, train_time, test_time = _run_inprocess(clf, X_train, y_train, X_test, y_test)

    print(f"  [clf] {clf_name} — accuracy={acc:.4f}, train={train_time}s, test={test_time}s")
    return acc, train_time, test_time


def _run_inprocess(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unsafe cast", module="numba")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)

        t0 = time.time()
        y_pred = clf.predict(X_test)
        test_time = round(time.time() - t0, 2)

    acc = accuracy_score(y_test, y_pred)
    return acc, train_time, test_time


def _run_lite_in_subprocess(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    q = Queue()
    p = Process(target=_lite_worker, args=(clf, X_train, y_train, X_test, y_test, q))
    p.start()
    result = q.get()
    p.join()

    if isinstance(result, Exception):
        raise result
    return result


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def append_result(row: dict, output_file: str) -> None:
    """Append a single result row to a CSV, writing the header only on first write."""
    write_header = not os.path.exists(output_file)
    pd.DataFrame([row]).to_csv(output_file, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    datasets: list[str],
    classifiers_factory,
    reduction_methods: dict,
    retention_rates: list[float],
    output_file: str,
    neighborhood_file: str,
    neighborhood_ks: list[int] = None,
) -> pd.DataFrame:
    """
    For each dataset × (original + method × retention_rate) × classifier:
    reduce, evaluate neighborhood preservation (once per method/rate/k), train,
    evaluate accuracy, and persist both result sets incrementally.

    Parameters
    ----------
    datasets : list of aeon dataset names
    classifiers_factory : callable() -> dict[str, classifier]
        Called fresh per combination to avoid state bleed between runs.
    reduction_methods : dict[str, callable]
    retention_rates : list of floats in (0, 1] — fraction of timepoints to keep
    output_file : path to the classification results CSV
    neighborhood_file : path to the neighborhood preservation results CSV
    neighborhood_ks : list of k values for precision@k and trustworthiness (default [5])
    """
    if neighborhood_ks is None:
        neighborhood_ks = [5]
    from .datasets import load_and_normalize
    from .metrics import compute_neighborhood_metrics

    for dataset in datasets:
        try:
            X_train, y_train, X_test, y_test = load_and_normalize(dataset)

            # --- Original (no reduction) ---
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset} | Original ({X_train.shape[2]} timepoints)")
            print(f"{'='*60}")
            for clf_name, clf in classifiers_factory().items():
                acc, train_t, test_t = train_and_evaluate(
                    clf_name, clf, X_train, y_train, X_test, y_test
                )
                append_result(
                    {
                        "dataset": dataset,
                        "classifier": clf_name,
                        "reduction_method": None,
                        "retention_rate": None,
                        "series_size": X_train.shape[2],
                        "accuracy": acc,
                        "train_time_s": train_t,
                        "test_time_s": test_t,
                        "reduction_time_s": 0,
                    },
                    output_file,
                )

            # --- Reduced ---
            for method_name, method in reduction_methods.items():
                for rate in retention_rates:
                    print(f"\n{'='*60}")
                    print(f"Dataset: {dataset} | {method_name} | retention={rate}")
                    print(f"{'='*60}")
                    try:
                        X_tr, X_te, red_t = reduce_dataset(
                            method_name, method, X_train, X_test, rate
                        )
                    except Exception as e:
                        print(f"[error] Reduction failed: {e}")
                        continue

                    # Neighborhood preservation — computed once per (method, rate, k)
                    print(f"  [metrics] Computing neighborhood preservation (ks={neighborhood_ks})...")
                    for k in neighborhood_ks:
                        try:
                            nb_metrics = compute_neighborhood_metrics(X_test, X_te, k=k)
                            append_result(
                                {
                                    "dataset": dataset,
                                    "reduction_method": method_name,
                                    "retention_rate": rate,
                                    "series_size": X_tr.shape[2],
                                    "k": k,
                                    **nb_metrics,
                                },
                                neighborhood_file,
                            )
                            print(f"  [metrics] k={k} | " + " | ".join(f"{m}={v:.4f}" for m, v in nb_metrics.items()))
                        except Exception as e:
                            print(f"  [metrics] k={k} Failed: {e}")

                    # Classification — one row per classifier
                    for clf_name, clf in classifiers_factory().items():
                        try:
                            acc, train_t, test_t = train_and_evaluate(
                                clf_name, clf, X_tr, y_train, X_te, y_test
                            )
                        except Exception as e:
                            print(f"[error] Classifier {clf_name} failed: {e}")
                            acc, train_t, test_t = float("nan"), float("nan"), float("nan")

                        append_result(
                            {
                                "dataset": dataset,
                                "classifier": clf_name,
                                "reduction_method": method_name,
                                "retention_rate": rate,
                                "series_size": X_tr.shape[2],
                                "accuracy": acc,
                                "train_time_s": train_t,
                                "test_time_s": test_t,
                                "reduction_time_s": red_t,
                            },
                            output_file,
                        )

                    del X_tr, X_te
                    torch.cuda.empty_cache()
                    gc.collect()

            del X_train, X_test, y_train, y_test
            gc.collect()

        except Exception as e:
            print(f"[error] Dataset '{dataset}' failed: {e}")
            continue

    return pd.read_csv(output_file)
