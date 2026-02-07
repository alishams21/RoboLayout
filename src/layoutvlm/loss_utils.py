"""
Loss logging and plotting utilities for the gradient solver.
Centralizes: loss directory path, printing loss breakdown, and saving loss curve PNGs.
"""
import os
from typing import Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def get_loss_dir(save_dir: str) -> str:
    """Build the loss output directory path from the group save_dir (e.g. save_dir/../loss)."""
    return os.path.join(os.path.dirname(save_dir), "loss")


def ensure_loss_dir(loss_dir: str) -> None:
    """Create the loss directory if it does not exist."""
    os.makedirs(loss_dir, exist_ok=True)


def _scalar(x: Any) -> float:
    """Convert tensor or scalar to float for printing/plotting."""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def print_loss_breakdown(
    iteration: int,
    total: Any,
    overlap: Any,
    existing_constraint: Any,
    new_constraint: Any,
    reachability: Optional[Any] = None,
) -> None:
    """Print one line: iteration and Total, Overlap, Existing Constraint, New Constraint, optionally Reachability."""
    total_f = _scalar(total)
    overlap_f = _scalar(overlap)
    existing_f = _scalar(existing_constraint)
    new_f = _scalar(new_constraint)
    parts = [
        f"Iteration {iteration}, Total Loss: {total_f}",
        f"Overlap Loss: {overlap_f}",
        f"Existing Constraint Loss: {existing_f}",
        f"New Constraint Loss: {new_f}",
    ]
    if reachability is not None:
        parts.append(f"Reachability Loss: {_scalar(reachability)}")
    print(", ".join(parts))


def plot_loss_curves(
    loss_history: List[dict],
    plot_dir: str,
    filename_suffix: str = "0",
) -> str:
    """
    Plot loss curves from loss_history and save PNG under plot_dir.
    loss_history: list of dicts with keys 'total', 'overlap', 'existing_constraint', 'new_constraint',
                  and optionally 'reachability'. Values can be tensors or scalars.
    Returns the path of the saved file and prints a confirmation message.
    """
    ensure_loss_dir(plot_dir)
    out_path = os.path.join(plot_dir, f"loss_curves_{filename_suffix}.png")

    if not loss_history:
        return out_path

    iterations = list(range(len(loss_history)))
    total = [_scalar(h["total"]) for h in loss_history]
    overlap = [_scalar(h["overlap"]) for h in loss_history]
    existing = [_scalar(h["existing_constraint"]) for h in loss_history]
    new = [_scalar(h["new_constraint"]) for h in loss_history]
    has_reach = loss_history and "reachability" in loss_history[0] and loss_history[0]["reachability"] is not None
    reach = [_scalar(h.get("reachability") or 0) for h in loss_history] if has_reach else None

    fig, (ax, ax_log) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Linear-scale plot
    ax.plot(iterations, total, label="Total", color="black", linewidth=1.5)
    ax.plot(iterations, overlap, label="Overlap", alpha=0.8)
    ax.plot(iterations, existing, label="Existing Constraint", alpha=0.8)
    ax.plot(iterations, new, label="New Constraint", alpha=0.8)
    if reach is not None:
        ax.plot(iterations, reach, label="Reachability", alpha=0.8)
    ax.set_ylabel("Loss")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Loss curves (linear scale)")
    ax.grid(True, alpha=0.3)

    # Log-scale plot on |loss| to handle zeros/negatives
    eps = 1e-6
    total_log = np.log10(np.maximum(np.abs(total), eps))
    overlap_log = np.log10(np.maximum(np.abs(overlap), eps))
    existing_log = np.log10(np.maximum(np.abs(existing), eps))
    new_log = np.log10(np.maximum(np.abs(new), eps))
    ax_log.plot(iterations, total_log, label="log10 |Total|", color="black", linewidth=1.5)
    ax_log.plot(iterations, overlap_log, label="log10 |Overlap|", alpha=0.8)
    ax_log.plot(iterations, existing_log, label="log10 |Existing Constraint|", alpha=0.8)
    ax_log.plot(iterations, new_log, label="log10 |New Constraint|", alpha=0.8)
    if reach is not None:
        reach_log = np.log10(np.maximum(np.abs(reach), eps))
        ax_log.plot(iterations, reach_log, label="log10 |Reachability|", alpha=0.8)
    ax_log.set_xlabel("Iteration")
    ax_log.set_ylabel("log10 |Loss|")
    ax_log.legend(loc="best", fontsize=8)
    ax_log.set_title("Loss curves (log scale)")
    ax_log.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Loss curves saved to {out_path}")
    return out_path
