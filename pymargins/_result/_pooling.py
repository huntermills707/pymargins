"""Rubin pooling combinator for multiple-imputation results."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ._graphresult import GraphResult
from ._margins import MarginsResult
from ._scales import _phi_to_name

# ---------------------------------------------------------------------------
# Diagnostic carrier
# ---------------------------------------------------------------------------


@dataclass
class ImputationDiagnostic:
    """Diagnostic information from Rubin pooling.

    Each per-component field is a Python ``float`` for a scalar estimand and an
    ``np.ndarray`` (shaped like ``MarginsResult.estimate``) for a vector one.

    Attributes
    ----------
    n_imputations : int
        Number of imputations pooled.
    fmi : np.ndarray | float
        Fraction of missing information per component.
    relative_efficiency : np.ndarray | float
        Relative efficiency per component.
    df : np.ndarray | float
        Degrees of freedom per component.
    within_var : np.ndarray | float
        Within-imputation variance per component.
    between_var : np.ndarray | float
        Between-imputation variance per component.
    total_var : np.ndarray | float
        Total variance per component.
    riv : np.ndarray | float
        Relative increase in variance per component.
    """

    n_imputations: int
    fmi: np.ndarray | float
    relative_efficiency: np.ndarray | float
    df: np.ndarray | float
    within_var: np.ndarray | float
    between_var: np.ndarray | float
    total_var: np.ndarray | float
    riv: np.ndarray | float

    def footer(self) -> str:
        """Return a one-line diagnostic summary for summary() footers."""
        fmi = np.asarray(self.fmi)
        df = np.asarray(self.df)
        re = np.asarray(self.relative_efficiency)
        if fmi.ndim == 0:
            fmi_str = f"{float(fmi):.3f}"
            df_str = f"{float(df):.1f}"
            re_str = f"{float(re):.3f}"
        else:
            fmi_str = f"max={float(np.nanmax(fmi)):.3f}"
            df_str = f"min={float(np.nanmin(df)):.1f}"
            re_str = f"min={float(np.nanmin(re)):.3f}"
        return (
            f"MI pooled (Rubin): M={self.n_imputations}, FMI {fmi_str}, "
            f"df {df_str}, rel. eff. {re_str}"
        )


# ---------------------------------------------------------------------------
# Internal: Rubin arithmetic
# ---------------------------------------------------------------------------


def _rubin_pool(est_inf, var_inf, level, complete_df=None):
    """Per-component Rubin pooling.

    Parameters
    ----------
    est_inf : ndarray, shape (M, k)
        Estimates on the inference scale.
    var_inf : ndarray, shape (M, k)
        Variances (SE²) on the inference scale.
    level : float
        Confidence level.
    complete_df : float, optional
        Complete-data df for Barnard–Rubin correction.

    Returns
    -------
    dict
        Keys: qbar, W, B, T, se, riv, df, fmi, re, lo, hi.
    """
    est = np.asarray(est_inf, float)
    var = np.asarray(var_inf, float)
    M = est.shape[0]
    qbar = est.mean(0)
    W = var.mean(0)
    if np.any(W <= 0):
        raise ValueError("pool_imputations: within-imputation variance is zero.")
    B = np.clip(est.var(0, ddof=1), 0.0, None)
    T = W + (1.0 + 1.0 / M) * B
    zero = B <= 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        riv = np.where(zero, 0.0, (1.0 + 1.0 / M) * B / W)
        lam = np.where(zero, 0.0, (1.0 + 1.0 / M) * B / T)
        df_old = np.where(zero, np.inf, (M - 1) * (1.0 + 1.0 / riv) ** 2)
    if complete_df is not None:
        nu = float(complete_df)
        # df_old is +inf wherever B<=0; the np.where discards those positions
        # (selecting nu), but the inf/inf division still warns unless silenced.
        with np.errstate(divide="ignore", invalid="ignore"):
            df_obs = (nu + 1.0) / (nu + 3.0) * nu * (1.0 - lam)
            df = np.where(zero, nu, df_old * df_obs / (df_old + df_obs))
    else:
        df = df_old
    with np.errstate(divide="ignore", invalid="ignore"):
        fmi = np.where(zero, 0.0, (riv + 2.0 / (df + 3.0)) / (riv + 1.0))
    re = 1.0 / (1.0 + fmi / M)
    se = np.sqrt(T)
    tcrit = stats.t.ppf(0.5 + level / 2.0, df)  # df=inf → normal quantile
    return dict(
        qbar=qbar,
        W=W,
        B=B,
        T=T,
        se=se,
        riv=riv,
        df=df,
        fmi=fmi,
        re=re,
        lo=qbar - tcrit * se,
        hi=qbar + tcrit * se,
    )


# ---------------------------------------------------------------------------
# Internal: commensurability validation
# ---------------------------------------------------------------------------


# Metadata keys that jointly identify the estimand. All M imputations must
# agree on every one of these: pooling assumes the same estimand computed
# under a shared posture across imputations (design §6.3, consideration 3).
_ESTIMAND_KEYS = ("labels", "kind", "at", "over", "scenarios")


def _meta_equal(a: object, b: object) -> bool:
    """Structural equality tolerant of arrays / frames nested in metadata.

    Estimand metadata can nest numpy arrays or pandas frames inside dicts and
    lists (e.g. ``scenarios``), where a bare ``==`` raises "truth value is
    ambiguous". This compares recursively and degrades gracefully to ``False``
    on any value it cannot compare, so an unrecognised structure is treated as
    a mismatch rather than crashing the validator.
    """
    if a is b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_meta_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _meta_equal(x, y) for x, y in zip(a, b, strict=False)
        )
    if hasattr(a, "equals") and hasattr(b, "equals"):  # pandas Frame/Series
        try:
            return bool(a.equals(b))
        except Exception:
            return False
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(a, b))
        except Exception:
            return False
    try:
        return bool(a == b)
    except (ValueError, TypeError):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:
            return False


def _validate_commensurable(results: list) -> type:
    """Validate that all results can be pooled together.

    Duck-typed: accepts any object exposing ``estimate/std_error/phi/phi_inv/
    level/method/estimand_metadata``. Returns the input result class so the
    pooled object matches the input type.
    """
    if len(results) < 2:
        raise ValueError("pool_imputations requires at least two results.")

    ref = results[0]
    ref_cls = type(ref)
    ref_shape = np.asarray(ref.estimate).shape
    ref_level = ref.level
    ref_phi_name = _phi_to_name(ref.phi)
    ref_phi_inv_name = _phi_to_name(ref.phi_inv)

    # Soft warning: same session reused (only legacy MarginsResult carries session)
    first_session = getattr(ref, "session", None)
    if first_session is not None:
        same_session = True
        for r in results[1:]:
            if getattr(r, "session", None) is not first_session:
                same_session = False
                break
        if same_session:
            warnings.warn(
                "All results appear to come from the same session. "
                "Pooling re-uses of one fit yields B≈0 and no MI value.",
                UserWarning,
                stacklevel=3,
            )

    for i, r in enumerate(results):
        if type(r) is not ref_cls:
            raise ValueError(
                f"pool_imputations: result {i} is {type(r).__name__}, expected "
                f"{ref_cls.__name__}. All results must have the same type."
            )

        # Shape
        shape = np.asarray(r.estimate).shape
        if shape != ref_shape:
            raise ValueError(
                f"pool_imputations: result {i} has shape {shape}, expected {ref_shape}."
            )

        # Estimand identity: labels, kind, at, over, scenarios must all agree.
        for key in _ESTIMAND_KEYS:
            got = r.estimand_metadata.get(key)
            want = ref.estimand_metadata.get(key)
            if not _meta_equal(got, want):
                raise ValueError(
                    f"pool_imputations: result {i} has {key}={got!r}, expected "
                    f"{want!r}. All results must be the same estimand computed "
                    "under a shared posture across imputations."
                )

        # Level
        if r.level != ref_level:
            raise ValueError(
                f"pool_imputations: result {i} has level={r.level}, expected {ref_level}."
            )

        # Scale (identity first, then name map)
        phi_name = _phi_to_name(r.phi)
        phi_inv_name = _phi_to_name(r.phi_inv)
        phi_match = (
            (phi_name == ref_phi_name)
            if phi_name is not None and ref_phi_name is not None
            else (r.phi is ref.phi)
        )
        phi_inv_match = (
            (phi_inv_name == ref_phi_inv_name)
            if phi_inv_name is not None and ref_phi_inv_name is not None
            else (r.phi_inv is ref.phi_inv)
        )
        if not phi_match or not phi_inv_match:
            raise ValueError(
                f"pool_imputations: result {i} has incompatible scale (phi/phi_inv). "
                "All results must use the same inference scale."
            )

        # Finite / non-negative
        est_arr = np.asarray(r.estimate)
        se_arr = np.asarray(r.std_error)
        if not np.all(np.isfinite(est_arr)):
            raise ValueError(f"pool_imputations: result {i} has non-finite estimates.")
        if not np.all(np.isfinite(se_arr)):
            raise ValueError(
                f"pool_imputations: result {i} has non-finite standard errors."
            )
        if np.any(se_arr < 0):
            raise ValueError(
                f"pool_imputations: result {i} has negative standard errors."
            )

    return ref_cls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pool_imputations(results, *, label="pooled", complete_df=None):
    """Pool result objects across imputations via Rubin's rules.

    All results must be the same estimand (matching labels) on imputed
    copies of the same data, computed under a shared posture. Pools on
    the inference scale; reports FMI / relative efficiency / df.

    Parameters
    ----------
    results : list
        One result per imputation. During the R4-R6 window both
        ``MarginsResult`` and ``GraphResult`` are accepted; the returned type
        matches the input type.
    label : str, default "pooled"
        Provenance tag for the pooled estimand, stored on the result as
        ``estimand_metadata["pooled_label"]`` for downstream bookkeeping. It
        does not rename the per-component row labels (those carry over from the
        input results) and is not shown in ``summary()``.
    complete_df : float, optional
        Complete-data degrees of freedom for the Barnard–Rubin small-sample
        correction. If None, uses the classic Rubin (1987) df formula.

    Returns
    -------
    MarginsResult or GraphResult
        A leaf result with ``method="pooled"`` and ``imputation_diagnostic`` set.
    """
    ref_cls = _validate_commensurable(results)
    ref = results[0]
    phi, phi_inv, level = ref.phi, ref.phi_inv, ref.level

    est_inf = np.stack(
        [
            np.atleast_1d(
                phi_inv(r.estimate)
                if phi_inv is not None
                else np.asarray(r.estimate, float)
            )
            for r in results
        ]
    )
    var_inf = np.stack(
        [np.atleast_1d(np.asarray(r.std_error, float)) ** 2 for r in results]
    )
    p = _rubin_pool(est_inf, var_inf, level, complete_df)

    rep = (
        (lambda a: np.asarray(phi(a))) if phi is not None else (lambda a: np.asarray(a))
    )
    scalar = np.asarray(ref.estimate).ndim == 0
    sq = (lambda a: float(np.reshape(a, ()))) if scalar else (lambda a: np.asarray(a))
    _sq_arr = (
        (lambda a: float(np.reshape(a, ()))) if scalar else (lambda a: np.asarray(a))
    )
    diag = ImputationDiagnostic(
        n_imputations=len(results),
        fmi=_sq_arr(p["fmi"]),
        relative_efficiency=_sq_arr(p["re"]),
        df=_sq_arr(p["df"]),
        within_var=_sq_arr(p["W"]),
        between_var=_sq_arr(p["B"]),
        total_var=_sq_arr(p["T"]),
        riv=_sq_arr(p["riv"]),
    )
    meta = {**ref.estimand_metadata, "pooled_label": label}

    kwargs = dict(
        estimate=sq(rep(p["qbar"])),
        std_error=sq(p["se"]),
        conf_int_lower=sq(rep(p["lo"])),
        conf_int_upper=sq(rep(p["hi"])),
        method="pooled",
        level=level,
        n_obs=max(r.n_obs for r in results),
        estimand_metadata=meta,
        phi=phi,
        phi_inv=phi_inv,
        imputation_diagnostic=diag,
    )

    if ref_cls is GraphResult:
        return GraphResult(
            **kwargs,
            labels=meta.get("labels"),
            ci="wald",
            scale=ref.scale if hasattr(ref, "scale") else "response",
            at=ref.at if hasattr(ref, "at") else "overall",
            plan=ref.plan if hasattr(ref, "plan") else None,
            population_note=ref.population_note if hasattr(ref, "population_note") else None,
        )
    return MarginsResult(**kwargs, session=None)
