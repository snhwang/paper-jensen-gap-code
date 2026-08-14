"""Minimal b-value shell clustering.

Extracted standalone from the lab's diffusion processor so the gap-computation
scripts have no heavy dependency. Groups unique diffusion-weighted b-values that
lie within SHELL_TOLERANCE of each other into nominal shells (e.g. b~1500,
b~3000), returning (nominal_bval, volume_mask) for each shell.
"""
import numpy as np

SHELL_TOLERANCE = 100  # b-values within this gap are considered the same shell


def cluster_bvals_into_shells(bvals):
    """Cluster b-values into nominal shells.

    Returns a list of (nominal_bval, mask_over_all_volumes), excluding b<=50
    (treated as b0).
    """
    diff_bvals = bvals[bvals > 50]
    if len(diff_bvals) == 0:
        return []

    sorted_unique = np.sort(np.unique(diff_bvals))
    shells = []
    current_group = [sorted_unique[0]]
    for bv in sorted_unique[1:]:
        if bv - current_group[-1] < SHELL_TOLERANCE:
            current_group.append(bv)
        else:
            shells.append(current_group)
            current_group = [bv]
    shells.append(current_group)

    result = []
    for group in shells:
        nominal = int(round(np.mean(group) / 50) * 50)
        mask = np.zeros(len(bvals), dtype=bool)
        for bv in group:
            mask |= (bvals == bv)
        result.append((nominal, mask))
    return result
