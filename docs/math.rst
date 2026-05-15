Mathematical motivation
=======================

This page derives, in one place, the statistics
:class:`~pymargins.Margins` computes and the uncertainty attached to
each. The motivation matches Stata's `margins-delta-method FAQ
<https://www.stata.com/support/faqs/statistics/compute-standard-errors-with-margins/>`_
and Richard Williams' `Margins01 notes
<https://academicweb.nd.edu/~rwilliam/stats/Margins01.pdf>`_; the goal
here is to expose the *single Jacobian primitive* the implementation
reduces to, and to write down the curvature diagnostic that decides
when the delta method is unsafe.

The delta method
----------------

For a fitted parameter vector :math:`\hat\beta` with estimated
covariance :math:`\widehat V`, and a (possibly vector-valued) statistic
:math:`g(\beta)`, a first-order Taylor expansion gives

.. math::

    g(\hat\beta) \;\approx\; g(\beta_0)
        + G\,(\hat\beta - \beta_0),
    \qquad
    G = \left.\tfrac{\partial g}{\partial \beta}\right|_{\beta_0}.

Taking variances yields

.. math::

    \widehat{\operatorname{Var}}\bigl[g(\hat\beta)\bigr]
        \;\approx\; G\,\widehat V\,G^\top .

Three things to notice:

1. The approximation depends only on **first** derivatives of :math:`g`.
2. Once :math:`G\widehat V G^\top` is in hand, *any* linear combination
   :math:`C\,g(\hat\beta)` has covariance :math:`C(G\widehat V G^\top)C^\top`
   — no further differentiation needed. That is why
   :meth:`~pymargins.Margins.contrasts` is exact under the same
   approximation.
3. ``pymargins`` computes :math:`G` by JAX autodiff when an autodiff
   path exists for the predict function, by autodiff over a custom-JVP
   FD primitive when the model is black-box but has a clean
   :math:`\eta = X\beta` linear predictor, and by full finite
   differences otherwise. See
   :doc:`explanations/gradient_backend`.

A single estimand schema
------------------------

Internal to the library every estimand is a triple
:math:`(h, \phi, \phi^{-1})`:

- :math:`h(\beta)` — the estimand on the **inference scale**; this is
  what gets differentiated for delta and what gets evaluated for
  simulation.
- :math:`\phi` — back-transform from inference scale to reporting
  scale, applied to CI endpoints.
- :math:`\phi^{-1}` — forward transform; converts user-supplied null
  values onto the inference scale for hypothesis tests.

The pair :math:`(\phi, \phi^{-1})` is **session-level**, not
per-estimand: every call within one :class:`~pymargins.Margins`
instance is on the same inference scale. That is what makes the
session-level κ diagnostic and inter-call composability work.

The Williams (2012) statistic table reduces to a single primitive:

.. list-table::
    :header-rows: 1
    :widths: 15 60

    * - Statistic
      - :math:`g(\beta)`
    * - AAP
      - :math:`\tfrac{1}{n}\sum_i f(x_i^\top\beta)`
    * - APM
      - :math:`f(\bar x^\top\beta)`
    * - APR
      - :math:`\tfrac{1}{n}\sum_i f(x_i^\top\beta)` with some :math:`x` fixed
    * - AME
      - :math:`\tfrac{1}{n}\sum_i \partial f(x_i^\top\beta)/\partial x_{ij}`
    * - MEM
      - :math:`\partial f(\bar x^\top\beta)/\partial x_j`
    * - MER
      - AME with some :math:`x` held at representative values
    * - Contrast
      - :math:`E[f(\cdot)\mid x_j=\ell] - E[f(\cdot)\mid x_j=\text{ref}]`

Every entry is a linear combination of mean-predictions. The single
Jacobian primitive is

.. math::

    \frac{\partial}{\partial\beta}\,
    \frac{1}{n}\sum_i f(x_i^\top\beta)
    \;=\; \frac{1}{n}\sum_i f'(x_i^\top\beta)\,x_i .

Substituting back into the delta method gives a closed form for the
covariance of every statistic above.

Beyond delta: simulation and bootstrap
--------------------------------------

The delta method is a first-order approximation: :math:`G\widehat V
G^\top` is exact only to the extent that :math:`g` is locally linear
in :math:`\beta` near :math:`\hat\beta`. For statistics that are
sharply nonlinear (a probability near 0 or 1, an elasticity at a
near-zero prediction, a ratio with a small denominator) Wald
intervals can extend beyond the natural support or under-cover.

``pymargins`` exposes two alternatives behind the same ``method=``
keyword on the session.

**Krinsky–Robb simulation** (``method="simulation"``). Draw
:math:`S` parameter vectors :math:`\beta_s\sim N(\hat\beta,\widehat
V)`, evaluate :math:`g(\beta_s)`, and read inference off the
empirical distribution:

.. math::

    \widehat{\operatorname{Var}}_{\mathrm{KR}}\bigl[g(\hat\beta)\bigr]
    = \frac{1}{S-1}\sum_{s=1}^S
        \bigl(g(\beta_s) - \bar g\bigr)\bigl(g(\beta_s) - \bar g\bigr)^{\!\top}.

The reported point estimate stays the analytic :math:`g(\hat\beta)`
(not the Monte Carlo mean), matching Stata's ``vce(simulation)``.
Pointwise CIs default to empirical quantiles of the draws — so an
asymmetric CI for an extreme probability is natural.

**Bootstrap** (``method="bootstrap"``). Refit the model on :math:`B`
resamples and read inference off the bootstrap distribution. Three
resampling schemes:

- pairs — :math:`(y_i, x_i)` resampled IID (default);
- cluster — whole clusters resampled, required for within-cluster
  correlation;
- block — moving-block resampling for time series.

Failed refits are caught and counted; a ``RuntimeWarning`` fires when
the failure rate exceeds 5%.

The κ curvature diagnostic
--------------------------

``pymargins`` computes Skovgaard's relative curvature κ for every
estimand (when diagnostics are enabled) and auto-falls-back to
simulation when κ exceeds the session threshold (default 0.3). This
is a meaningful divergence from Stata's ``margins`` and
``marginaleffects``, both of which always do delta and never tell you
when delta is suspect.

Heuristically, κ is the ratio of the second-derivative contribution
of :math:`g` (Hessian, whitened by :math:`\widehat V`) to its
first-derivative contribution (gradient, same whitening). When κ is
small the linearization underlying the delta method is accurate;
when κ is large the symmetric Wald interval can miss the true
sampling distribution badly. The thresholds (0.1 / 0.3) are
calibrated from the nonlinear-regression literature; expose them as
configurable but keep these as defaults.

See :doc:`explanations/kappa_diagnostic` for the whitening transform
and the auto-fallback policy.

Inference scales (``phi``) and the chain rule
---------------------------------------------

The :class:`~pymargins.Margins` constructor commits to a
:math:`(\phi, \phi^{-1})` pair via either the classmethod helpers
(``Margins.log_scale``, ``Margins.logit_scale``,
``Margins.correlation_scale``,
``Margins.linear_scale``) or by passing ``phi=`` and ``phi_inv=``
directly.

The chain rule fixes how :math:`\phi` propagates through the
estimand:

- For a single prediction :math:`g(\eta) = \phi(f(X\beta))`,
  :math:`\partial g/\partial\beta = \phi'(f(\eta))\,f'(\eta)\,X`.
- For an AME on the :math:`\phi`-scale,
  :math:`\mathrm{AME}_k^{(\phi)} = \tfrac{1}{n}\sum_i \phi'(f(\eta_i))\,f'(\eta_i)\,\beta_k`,
  whose Jacobian w.r.t. :math:`\beta` carries both a curvature term
  (from differentiating :math:`\phi'f'` through :math:`\eta`) and a
  level term (from the explicit :math:`\beta_k`).

In practice you do not write the chain rule yourself: JAX does it for
you. The session contract is just that you commit to one
:math:`(\phi, \phi^{-1})` for the whole analysis.

Why the response scale matters for DiD (Ai & Norton 2003)
---------------------------------------------------------

In a logit, the coefficient on a ``group:condition`` interaction is on
the *log-odds* scale. On the *probability* scale, the
difference-in-differences

.. math::

    \mathrm{DiD}(x_*)
        = \bigl[f(\eta_{1,1}) - f(\eta_{1,0})\bigr]
        - \bigl[f(\eta_{0,1}) - f(\eta_{0,0})\bigr]

is a nonlinear function of every parameter and every covariate
profile :math:`x_*`. You cannot read it off the interaction
coefficient. The right tool is
:meth:`~pymargins.Margins.contrasts` (or the
:func:`~pymargins.did` scenario helper), which evaluates the four
cells on the response scale with their joint delta-method covariance
and forms the DiD as a contrast.

References
----------

- Stata FAQ, *How are the standard errors computed with margins?*
  https://www.stata.com/support/faqs/statistics/compute-standard-errors-with-margins/
- Williams, R. (2012). *Using the margins command to estimate and
  interpret adjusted predictions and marginal effects.* Stata Journal,
  12(2), 308–331.
- Ai, C., & Norton, E. C. (2003). *Interaction terms in logit and
  probit models.* Economics Letters, 80(1), 123–129.
- Skovgaard, I. M. (1985). *A second-order investigation of asymptotic
  ancillarity.* Annals of Statistics, 13(2), 534–551.
- Krinsky, I., & Robb, A. L. (1986). *On approximating the
  statistical properties of elasticities.* Review of Economics and
  Statistics, 68(4), 715–719.
