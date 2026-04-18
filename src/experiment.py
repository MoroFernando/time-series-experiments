"""
Core experiment logic: dimensionality reduction, classifier training, and result persistence.
"""
import gc
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string (e.g. '2h 03m 45s')."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _print_clf_status(clf_name: str, idx: int, total: int, status: str) -> None:
    """
    Overwrite the current line with a classifier progress indicator.

    Uses a dot-based style (▷ ··· [N/M]) to visually differ from the
    block-bar used for reduction/training progress.
    """
    dots = "·" * 20
    label = f"{clf_name:<18}"
    sys.stdout.write(f"\r  ▷  {label} {dots}  [{idx}/{total}]  {status}      ")
    sys.stdout.flush()


def _print_clf_done(
    clf_name: str,
    idx: int,
    total: int,
    acc: float,
    train_t: float,
    test_t: float,
) -> None:
    """Print a completed-classifier line ending with a newline."""
    label = f"{clf_name:<18}"
    sys.stdout.write(
        f"\r  ✓  {label} acc={acc:.4f}  train={train_t}s  pred={test_t}s  [{idx}/{total}]\n"
    )
    sys.stdout.flush()


def _print_clf_error(clf_name: str, idx: int, total: int, err: Exception) -> None:
    """Print a failed-classifier line ending with a newline."""
    label = f"{clf_name:<18}"
    sys.stdout.write(
        f"\r  ✗  {label} FAILED: {err}  [{idx}/{total}]\n"
    )
    sys.stdout.flush()


def _print_eta(
    completed: int,
    total: int,
    durations: list[float],
    wall_start: float,
) -> None:
    """
    Print an overall experiment ETA line after each combination completes.

    Uses the mean duration of completed combinations as the per-combo estimate.
    """
    elapsed = time.time() - wall_start
    remaining = total - completed
    if completed > 0 and remaining > 0:
        avg = elapsed / completed
        eta_s = avg * remaining
        eta_str = _format_duration(eta_s)
    else:
        eta_str = "—"
    elapsed_str = _format_duration(elapsed)
    pct = completed / total if total else 0
    print(
        f"  ⏱  Combination {completed}/{total} done  |  "
        f"elapsed={elapsed_str}  |  ETA≈{eta_str}  ({pct:.0%})"
    )


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
        if hasattr(method, "cleanup"):
            method.cleanup()
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
        import warnings
        import time

        import torch
        if torch.cuda.is_available():
            torch.cuda.init()

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
    clf_idx: int = 0,
    clf_total: int = 0,
) -> tuple[float, float, float]:
    """
    Fit a classifier and evaluate it.

    LITE is run in an isolated subprocess so TensorFlow releases all GPU/RAM
    memory when the process exits. All other classifiers run in-process.

    Parameters
    ----------
    clf_idx / clf_total : position in the classifier loop (for progress display).
        Pass 0 for both to suppress the indexed display.

    Returns
    -------
    accuracy : float
    train_time_s : wall-clock seconds for clf.fit()
    test_time_s : wall-clock seconds for clf.predict()
    """
    idx, total = clf_idx, clf_total
    _print_clf_status(clf_name, idx, total, "fitting …")

    if clf_name == "LITE":
        acc, train_time, test_time = _run_lite_in_subprocess(clf, X_train, y_train, X_test, y_test)
    else:
        acc, train_time, test_time = _run_inprocess(clf, X_train, y_train, X_test, y_test)

    _print_clf_done(clf_name, idx, total, acc, train_time, test_time)
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
    # Use 'spawn' explicitly so the child process starts fresh without
    # inheriting the parent's CUDA context (avoids "Cannot re-initialize
    # CUDA in forked subprocess" when the default start method is 'fork').
    ctx = get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_lite_worker, args=(clf, X_train, y_train, X_test, y_test, q))
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

    # ------------------------------------------------------------------
    # Pre-compute total number of (dataset × combination) pairs for ETA.
    # A "combination" is one classifier batch: either the original run or
    # one (method, retention_rate) reduced run.
    # ------------------------------------------------------------------
    total_combos = len(datasets) * (1 + len(reduction_methods) * len(retention_rates))
    completed_combos = 0
    wall_start = time.time()

    for dataset in datasets:
        try:
            X_train, y_train, X_test, y_test = load_and_normalize(dataset)

            # --- Original (no reduction) ---
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset} | Original ({X_train.shape[2]} timepoints)")
            print(f"{'='*60}")
            clfs = classifiers_factory()
            clf_total = len(clfs)
            for clf_idx, (clf_name, clf) in enumerate(clfs.items(), start=1):
                acc, train_t, test_t = train_and_evaluate(
                    clf_name, clf, X_train, y_train, X_test, y_test,
                    clf_idx=clf_idx, clf_total=clf_total,
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
            completed_combos += 1
            _print_eta(completed_combos, total_combos, [], wall_start)

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
                        completed_combos += 1
                        _print_eta(completed_combos, total_combos, [], wall_start)
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
                    clfs = classifiers_factory()
                    clf_total = len(clfs)
                    for clf_idx, (clf_name, clf) in enumerate(clfs.items(), start=1):
                        try:
                            acc, train_t, test_t = train_and_evaluate(
                                clf_name, clf, X_tr, y_train, X_te, y_test,
                                clf_idx=clf_idx, clf_total=clf_total,
                            )
                        except Exception as e:
                            _print_clf_error(clf_name, clf_idx, clf_total, e)
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

                    completed_combos += 1
                    _print_eta(completed_combos, total_combos, [], wall_start)

            del X_train, X_test, y_train, y_test
            gc.collect()

        except Exception as e:
            print(f"[error] Dataset '{dataset}' failed: {e}")
            continue

    elapsed_total = time.time() - wall_start
    print(f"\n{'='*60}")
    print(f"Experiment complete — total time: {_format_duration(elapsed_total)}")
    print(f"{'='*60}")
    return pd.read_csv(output_file)
