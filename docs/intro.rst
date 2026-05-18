Introduction
============

``pymargins`` computes marginal-effects-style quantities — adjusted
predictions, slopes, contrasts, difference-in-differences, custom
linear and nonlinear combinations of predictions — from fitted
statistical models. Uncertainty is quantified via the delta method,
parametric simulation (Krinsky–Robb), or nonparametric bootstrap.

The design target is `Stata's
<https://www.stata.com/manuals/rmargins.pdf>`_ ``margins`` command and
the R package `marginaleffects
<https://marginaleffects.com/>`_, with one substantive difference:
:class:`~pymargins.Margins` is a **session** — once constructed it
commits to an inference scale, variance estimator, confidence level,
default evaluation point, and inference method. Every subsequent
computation inherits those commitments. Switching any of them requires
a new session.

Why marginal effects?
---------------------

A fitted nonlinear model reports on the wrong scale. A logistic
regression returns coefficients in log-odds; a Poisson model, in
log-counts; a Cox model, in log-hazards. The question an analyst or
stakeholder actually has — *how much does the predicted probability,
count, or survival move?* — is a marginal effect, and in a nonlinear
model that quantity is **not** any single coefficient. It is a
derivative or discrete contrast that

- depends on *where* in covariate space it is evaluated (the effect of
  ``bmi`` in a logit differs along the curve),
- combines several coefficients whenever the model contains
  interactions, polynomials, or splines, and
- carries its own standard error, which hand-calculation usually gets
  wrong.

``pymargins`` computes that quantity, on the outcome scale, with
uncertainty from the delta method, simulation, or bootstrap:

.. code-block:: python

    m = Margins.log_scale(fit, vcov="HC3")

    m.dydx("bmi")                            # AME on P(diabetes)
    m.predict(atexog={"age": [25, 45, 65]})  # adjusted predictions
    m.contrasts(scenarios=[...], contrasts=[+1, -1])  # risk difference

Reach for marginal effects when the model is nonlinear, when effects
are conditional (interactions / polynomials / splines), when the
audience needs outcome units rather than link-scale coefficients, when
heterogeneity across subgroups *is* the research question, or when the
answer is a counterfactual contrast ("what if everyone were
treated?"). These correspond exactly to the estimand axis below:
:meth:`~pymargins.Margins.predict`,
:meth:`~pymargins.Margins.dydx`, and
:meth:`~pymargins.Margins.contrasts` /
:meth:`~pymargins.Margins.evaluate`.

Why a session?
--------------

A session forces the analyst to declare their analytical posture
explicitly, in code, at one location. A reviewer sees the entire
methodological commitment in the constructor call. Any change in
posture — different scale, different vcov, different level — shows up
as a new ``Margins(...)`` in the audit trail.

The alternative (per-call configuration) is more flexible but invites
the analyst to try multiple scales, multiple vcov flavors, multiple
levels until something looks significant. ``pymargins`` makes this
*possible but visible*: do it, and you'll have multiple sessions in
your code, each documenting its own posture.

Installation
------------

.. code-block:: bash

    pip install pymargins

Requires Python ≥3.10 and JAX. Dependencies (``numpy``, ``pandas``,
``scipy``, ``jax``) are installed automatically; model-specific
backends (``statsmodels``, ``lifelines``, ``linearmodels``) are
detected at runtime when present.

Quickstart
----------

.. code-block:: python

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from pymargins import Margins

    fit = smf.glm(
        "diabetes ~ C(black) + C(female) + C(agegrp) + bmi + age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # Open a session — log scale, HC3 robust SEs, 95% CIs, delta method.
    m = Margins.log_scale(fit, vcov="HC3", level=0.95)
    print(m.summary())          # methods-section paragraph

    # Pre-flight: is delta reliable here?
    print(m.diagnose().summary())

    # Adjusted predictions at representative values.
    m.predict(atexog={"age": [25, 45, 65]})

    # Average marginal effects on the response scale.
    m.dydx("age")

    # A relative-risk contrast, two scenarios with a [+1, -1] weight.
    rr = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}, "label": "treated"},
            {"atexog": {"treatment": 0}, "label": "control"},
        ],
        contrasts=[+1, -1],
    )
    print(rr.summary())

Each call returns a :class:`~pymargins.MarginsResult` with
``.estimate``, ``.se``, ``.vcov``, ``.ci_lower``, ``.ci_upper``,
``.pvalue``, plus ``.summary()``, ``.to_frame()``, ``.to_latex()``,
and ``.to_html()``.

The three orthogonal axes
-------------------------

Every call is a point in a three-dimensional design space:

1. **Estimand** — what is computed? :meth:`~pymargins.Margins.predict`,
   :meth:`~pymargins.Margins.dydx`,
   :meth:`~pymargins.Margins.contrasts`, or
   :meth:`~pymargins.Margins.evaluate` (arbitrary differentiable
   composition).
2. **Aggregation** — where in covariate space?
   ``at="overall"`` (per-row, then average → AME / AAP),
   ``at="typical"`` / ``at="mean"`` (single representative point →
   MEM / APM), or ``atexog={...}`` (representative grid → MER / APR).
3. **Inference** — how is uncertainty quantified?
   ``method="delta"``, ``method="simulation"``, or ``method="bootstrap"``.

These are orthogonal. Any combination is meaningful. The session fixes
the scale, vcov, level, and default ``at`` / ``method``; per-call
keywords pick the estimand and override aggregation via ``atexog``.

What's supported
----------------

Out of the box, with adapter auto-detection:

- **statsmodels**: GLM (all families × links), OLS / WLS / GLS,
  discrete (Logit, Probit, Poisson, NegativeBinomial), MNLogit,
  ordered, GEE (Gaussian / nominal / ordinal), MixedLM, PHReg, RLM,
  QuantReg, zero-inflated.
- **linearmodels**: IV/2SLS/LIML/GMM, panel (FE/RE/BE),
  AbsorbingLS, FamaMacBeth.
- **lifelines**: CoxPHFitter, CoxTimeVaryingFitter, AAF, CRC splines,
  Weibull / LogLogistic / LogNormal / GeneralizedGamma /
  PiecewiseExponential AFTs.
- **scikit-learn**: Bootstrap-only inference for any estimator with
  ``fit()`` and ``predict()``.  Linear models should use statsmodels
  or linearmodels instead; see :doc:`tutorials/sklearn_models`.

Models fit without native formula support (array-fit statsmodels,
linearmodels, sklearn) can still use interactions, polynomials, and
splines by passing ``formula=`` and ``data=`` to ``Margins``; see
:doc:`howto/formula_interface`.

Adapters for other model classes can be registered via
:func:`~pymargins.register_adapter`; see
:doc:`howto/custom_adapter`.

Where to next
-------------

**Tutorials — learn by doing.**

- :doc:`tutorials/getting_started` — first session, AAP and AME.
- :doc:`tutorials/glm_logit`, :doc:`tutorials/glm_poisson`,
  :doc:`tutorials/ols_linear`, :doc:`tutorials/mnlogit` — GLM family
  walkthroughs.
- :doc:`tutorials/cox_survival`, :doc:`tutorials/aft_survival` —
  survival models.
- :doc:`tutorials/iv_2sls`, :doc:`tutorials/panel_fe` — instrumental
  variables and panel data.
- :doc:`tutorials/contrasts_and_did` — pairwise contrasts and DiD.
- :doc:`tutorials/inference_methods` — delta vs simulation vs
  bootstrap.
- :doc:`tutorials/scales_and_kappa` — picking an inference scale and
  reading the κ diagnostic.
- :doc:`tutorials/sklearn_models` — bootstrap inference for
  scikit-learn estimators.

**How-to guides — task-focused recipes.**

- Robust / cluster / block SEs:
  :doc:`howto/robust_clustered_ses`, :doc:`howto/bootstrap`,
  :doc:`howto/cluster_block_bootstrap`.
- Simultaneous CIs: :doc:`howto/simultaneous_ci`.
- Scenario builders: :doc:`howto/scenarios_helpers`,
  :doc:`howto/grid_predictions`, :doc:`howto/diff_in_diff`.
- Discrete / elasticities: :doc:`howto/elasticities`,
  :doc:`howto/discrete_changes`.
- κ fallback: :doc:`howto/kappa_fallback`.
- Extension and export: :doc:`howto/custom_adapter`,
  :doc:`howto/matching`, :doc:`howto/exporting_results`,
  :doc:`howto/plotting`.
- Formula support for array-fit models: :doc:`howto/formula_interface`.

**Explanations — theory and design.**

- :doc:`math` — delta method, statistic schema, scale chain rule.
- :doc:`explanations/session_precommitment` — why a session.
- :doc:`explanations/inference_scale` — ``phi`` / ``phi_inv``.
- :doc:`explanations/kappa_diagnostic` — Skovgaard's κ and the
  auto-fallback.
- :doc:`explanations/gradient_backend` — autodiff vs wrapped-FD vs FD.
- :doc:`explanations/delta_sim_bootstrap` — when each VCE is right.
- :doc:`explanations/scenarios_model` — ``at`` vs ``atexog``.
- :doc:`explanations/adapter_pattern` — how adapters fit a model.
- :doc:`explanations/vs_marginaleffects` — where we differ.

**Reference.**

- :doc:`api` — every public class and method.
- :doc:`demos` — the Williams (2012) walkthrough scripts.
