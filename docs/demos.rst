Replication scripts (archive)
=============================

The notebook-style dataset demos under
:doc:`Demos — end-to-end analyses <index>` are the recommended
entry point. The script-style demos archived here ship as standalone
``.py`` files in ``demo/`` and are kept for reference and exact
numerical reproduction against the Williams (2012) paper.

.. contents::
    :local:
    :depth: 1

.. _demo-williams:

Williams (2012) — Adjusted predictions and marginal effects
-----------------------------------------------------------

``demo/williams_2012_demo.py`` replicates, on a simulated NHANES-like
dataset, every core analysis from Richard Williams' Stata Journal
paper:

1. Logit model with factor variables
2. Adjusted Predictions at the Means (APM)
3. Average Adjusted Predictions (AAP)
4. Predictions at Representative Values (APR)
5. Marginal Effects at the Means (MEM) — continuous
6. Average Marginal Effects (AME) — continuous
7. Discrete changes for dummy variables
8. Marginal Effects at Representative Values (MER)
9. OLS model for comparison

.. literalinclude:: ../demo/williams_2012_demo.py
    :language: python
    :linenos:

.. _demo-williams-scales:

Williams (2012) — Inference scales
----------------------------------

``demo/williams_2012_demo_scales.py`` reruns the same analyses on
the log, logit, and lift inference scales, demonstrating how a
single session locks in :math:`(\phi, \phi^{-1})` for the whole
audit trail and how the κ diagnostic flags when the chosen scale is
poorly linearized.

.. literalinclude:: ../demo/williams_2012_demo_scales.py
    :language: python
    :linenos:

Cross-checks
^^^^^^^^^^^^

The companion files ``demo/williams_2012_demo_statsmodels.py``,
``demo/williams_2012_demo_marginaleffects_r.R``, and
``demo/williams_2012_demo_scales_marginaleffects_r.R`` reproduce the
same numbers using ``statsmodels.get_margeff`` and the R
``marginaleffects`` package. These exist to pin ``pymargins`` to
externally agreed answers at the precision both tools agree on.
