pymargins
=========

Expert-mode marginal effects for Python. Session-level analytical
pre-commitment, JAX-native autodiff, and a κ-driven simulation
fallback when the delta method is unsafe.

``pymargins`` wraps a fitted statistical model in a :class:`~pymargins.Margins`
session, then computes adjusted predictions, slopes, contrasts, and
arbitrary differentiable estimands — with uncertainty from the delta
method, Krinsky–Robb simulation, or bootstrap — across statsmodels,
linearmodels, and lifelines model classes.

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    intro

.. toctree::
    :maxdepth: 1
    :caption: Tutorials — learning by doing

    tutorials/getting_started
    tutorials/glm_logit
    tutorials/glm_poisson
    tutorials/ols_linear
    tutorials/mnlogit
    tutorials/cox_survival
    tutorials/aft_survival
    tutorials/iv_2sls
    tutorials/panel_fe
    tutorials/contrasts_and_did
    tutorials/inference_methods
    tutorials/scales_and_kappa

.. toctree::
    :maxdepth: 1
    :caption: How-to guides — task-focused recipes

    howto/robust_clustered_ses
    howto/bootstrap
    howto/cluster_block_bootstrap
    howto/simultaneous_ci
    howto/scenarios_helpers
    howto/grid_predictions
    howto/diff_in_diff
    howto/elasticities
    howto/discrete_changes
    howto/kappa_fallback
    howto/custom_adapter
    howto/matching
    howto/exporting_results
    howto/plotting
    howto/evaluate
    howto/contrasts
    howto/contrasts_vs_evaluate

.. toctree::
    :maxdepth: 2
    :caption: Reference

    api
    demos

.. toctree::
    :maxdepth: 1
    :caption: Explanations — theory and design

    math
    explanations/session_precommitment
    explanations/inference_scale
    explanations/kappa_diagnostic
    explanations/gradient_backend
    explanations/delta_sim_bootstrap
    explanations/scenarios_model
    explanations/adapter_pattern
    explanations/vs_marginaleffects

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
