"""
Post-optimization cleanup: identify problematic furniture (e.g. overlapping)
and re-optimize only those while keeping the rest fixed.
"""

from typing import Set, Dict, Any, List, Tuple

import torch
import os
from shapely.geometry import Polygon
from tqdm import tqdm

from utils.plot_utils import visualize_grid


def get_overlapping_asset_pairs(
    solver_assets: Dict[str, Any],
    is_fixture,
    on_top_of_assets: List[Tuple[str, str]],
) -> Set[Tuple[str, str]]:
    """
    Find pairs of assets whose 2D footprints overlap (excluding fixtures and on_top_of pairs).
    Returns set of (id_i, id_j) with i < j lexicographically.
    """
    overlapping_pairs = set()
    non_fixture_assets = [
        (instance_id, asset)
        for instance_id, asset in solver_assets.items()
        if not is_fixture(instance_id)
    ]
    n = len(non_fixture_assets)
    for i in range(n):
        id_i, asset_i = non_fixture_assets[i]
        try:
            corner_i = asset_i.get_2dpolygon().detach().cpu().numpy()
            poly_i = Polygon(corner_i)
            if not poly_i.is_valid:
                poly_i = poly_i.buffer(0)
        except Exception:
            continue
        for j in range(i + 1, n):
            id_j, asset_j = non_fixture_assets[j]
            if (id_i, id_j) in on_top_of_assets or (id_j, id_i) in on_top_of_assets:
                continue
            try:
                corner_j = asset_j.get_2dpolygon().detach().cpu().numpy()
                poly_j = Polygon(corner_j)
                if not poly_j.is_valid:
                    poly_j = poly_j.buffer(0)
                if poly_i.intersects(poly_j):
                    overlapping_pairs.add((id_i, id_j))
            except Exception:
                continue
    return overlapping_pairs


def identify_problematic_assets(
    grad_solver,
    solver_assets: Dict[str, Any],
) -> Set[str]:
    """
    Identify instance_ids that are involved in at least one overlapping pair.
    These will be re-optimized in the cleanup step while the rest are kept fixed.
    """
    overlapping_pairs = get_overlapping_asset_pairs(
        solver_assets,
        grad_solver.is_fixture,
        grad_solver.on_top_of_assets,
    )
    problematic = set()
    for (id_i, id_j) in overlapping_pairs:
        problematic.add(id_i)
        problematic.add(id_j)
    return problematic


def _freeze_non_problematic(
    grad_solver,
    solver_assets: Dict[str, Any],
    problematic_ids: Set[str],
):
    """Set optimize=0 and requires_grad=False for all assets not in problematic_ids."""
    original_optimize = {}
    original_requires_grad = {}
    for instance_id, asset in solver_assets.items():
        if grad_solver.is_fixture(instance_id):
            continue
        original_optimize[instance_id] = asset.optimize
        original_requires_grad[instance_id] = (
            asset.position.requires_grad,
            asset.rotation.requires_grad,
        )
        if instance_id in problematic_ids:
            asset.optimize = 1
            if not asset.position.requires_grad:
                asset.position.requires_grad_(True)
            if not asset.rotation.requires_grad:
                asset.rotation.requires_grad_(True)
        else:
            asset.optimize = 0
            asset.position.requires_grad_(False)
            asset.rotation.requires_grad_(False)
    return original_optimize, original_requires_grad


def _restore_optimize_and_grad(
    grad_solver,
    solver_assets: Dict[str, Any],
    original_optimize: Dict[str, int],
    original_requires_grad: Dict[str, Tuple[bool, bool]],
):
    """Restore original optimize and requires_grad for all assets."""
    for instance_id, asset in solver_assets.items():
        if grad_solver.is_fixture(instance_id):
            continue
        if instance_id in original_optimize:
            asset.optimize = original_optimize[instance_id]
        if instance_id in original_requires_grad:
            pos_grad, rot_grad = original_requires_grad[instance_id]
            asset.position.requires_grad_(pos_grad)
            asset.rotation.requires_grad_(rot_grad)


def run_cleanup_step(
    grad_solver,
    existing_constraints: List,
    new_constraints: List,
    iterations: int = 40,
    learning_rate: float = 0.01,
    temp_dir: str = None,
    project_interval: int = 10,
) -> None:
    """
    After main optimization: identify problematic furniture (e.g. overlapping),
    freeze the rest, and run a short gradient descent only on the problematic ones.
    Modifies grad_solver.solver_assets in place.
    """
    solver_assets = grad_solver.solver_assets
    all_constraints = existing_constraints + new_constraints

    problematic_ids = identify_problematic_assets(grad_solver, solver_assets)
    if not problematic_ids:
        return

    # Freeze all non-problematic assets (and record state to restore later)
    original_optimize, original_requires_grad = _freeze_non_problematic(
        grad_solver, solver_assets, problematic_ids
    )

    # Build parameter list only for problematic assets
    parameters = []
    for instance_id in problematic_ids:
        asset = solver_assets.get(instance_id)
        if asset is None or grad_solver.is_fixture(instance_id):
            continue
        if asset.optimize and asset.position.requires_grad:
            parameters.append(asset.position)
        if asset.optimize and asset.rotation.requires_grad:
            parameters.append(asset.rotation)
    if not parameters:
        _restore_optimize_and_grad(grad_solver, solver_assets, original_optimize, original_requires_grad)
        return

    previous_allow_nograd = getattr(grad_solver, "allow_nograd_constraints", False)
    grad_solver.allow_nograd_constraints = True
    try:
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for i in tqdm(range(iterations), desc="cleanup (problematic only)", leave=False):
            optimizer.zero_grad()
            overlap_loss, existing_constraint_loss, new_constraint_loss, reachability_loss = (
                grad_solver.calc_loss(existing_constraints, new_constraints)
            )
            loss = (
                overlap_loss
                + existing_constraint_loss
                + new_constraint_loss
                + reachability_loss
            ) * 0.01
            if not loss.requires_grad:
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            if (i + 1) % project_interval == 0:
                grad_solver.project_back_to_polygon(all_constraints)
            if temp_dir and (i % 10 == 0 or i == iterations - 1):
                frame_path = os.path.join(temp_dir, f"frame_cleanup_{i:04d}.png")
                visualize_grid(grad_solver.boundary, solver_assets, frame_path)
    finally:
        grad_solver.allow_nograd_constraints = previous_allow_nograd
        _restore_optimize_and_grad(grad_solver, solver_assets, original_optimize, original_requires_grad)