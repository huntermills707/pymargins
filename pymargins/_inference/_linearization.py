from __future__ import annotations

import numpy as np


def linearization_meat(
    weighted_scores: np.ndarray,  # u_i = w_i s_i, shape (n, p)
    psu: np.ndarray | None,  # shape (n,) or None
    strata: np.ndarray | None,  # shape (n,) or None
    fpc_fraction: np.ndarray | None,  # f_h per row, shape (n,) or None
    nest: bool = True,
) -> np.ndarray:
    """Design-based variance M of the weighted score total Σ_i u_i.

    Implements the stratified single-stage (PSU-level) Taylor linearization
    meat with the n_h/(n_h-1) finite-sample correction and optional FPC.
    Returns a (p, p) matrix.
    """
    n, p = weighted_scores.shape
    if strata is None:
        strata = np.zeros(n, dtype=int)
    if psu is None:
        psu = np.arange(n)  # each obs is its own PSU

    M = np.zeros((p, p))
    for h in np.unique(strata):
        in_h = strata == h
        psu_h = psu[in_h]
        u_h = weighted_scores[in_h]
        # PSU ids may repeat across strata; with nest=True scope them to h
        labels = psu_h  # already restricted to stratum rows
        uniq = np.unique(labels)
        n_h = len(uniq)
        if n_h < 2:
            raise ValueError(
                f"Stratum {h!r} has only {n_h} PSU(s) ('lonely PSU'); the "
                "n_h/(n_h-1) correction is undefined. Options: (1) collapse "
                "this stratum with another, (2) set the stratum's PSU id to "
                "match a PSU in a larger stratum, or (3) omit this stratum "
                "from the design. Rao-Wu rescaling or certainty-PSU "
                "approximations are not yet supported."
            )
        # Sum weighted scores within each PSU → (n_h, p)
        U = np.stack([u_h[labels == c].sum(axis=0) for c in uniq], axis=0)
        Ubar = U.mean(axis=0)
        D = U - Ubar
        # FPC: take the stratum's fraction (constant within stratum)
        f_h = 0.0
        if fpc_fraction is not None:
            f_h = float(np.asarray(fpc_fraction)[in_h][0])
        M += (1.0 - f_h) * (n_h / (n_h - 1.0)) * (D.T @ D)
    return M


def linearization_cov(
    bread: np.ndarray,  # A = non-robust cov_params(), (p, p)
    score_obs: np.ndarray,  # unweighted s_i, (n, p)
    weights: np.ndarray,  # w_i, (n,)
    psu: np.ndarray | None,
    strata: np.ndarray | None,
    fpc_fraction: np.ndarray | None,
    nest: bool = True,
) -> np.ndarray:
    """Full design-based sandwich V = A M Aᵀ."""
    u = weights[:, None] * np.asarray(score_obs)
    M = linearization_meat(u, psu, strata, fpc_fraction, nest)
    A = np.asarray(bread)
    return A @ M @ A.T
