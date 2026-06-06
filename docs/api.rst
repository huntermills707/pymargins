API reference
=============

.. automodule:: pymargins
    :no-members:
    :undoc-members:
    :show-inheritance:

.. currentmodule:: pymargins

Session
-------

.. autosummary::
    :toctree: _autosummary
    :recursive:

    Margins

Results
-------

.. autosummary::
    :toctree: _autosummary
    :recursive:

    MarginsResult
    TestResult
    DiagnosticResult

Scenario helpers
----------------

.. autosummary::
    :toctree: _autosummary

    pairwise
    reference
    at_levels
    grid
    did
    diff
    all_pairwise

Adapter interface
-----------------

.. autosummary::
    :toctree: _autosummary
    :recursive:

    ModelAdapter
    GLMAdapter
    LinearPredictionAdapter
    WrappedFDAdapter
    BootstrapOnlyAdapter
    VariableInfo
    register_adapter

Concrete adapters
-----------------

You rarely need these — ``Margins(model)`` auto-detects the right adapter
for standard statsmodels / linearmodels / lifelines results. Import a
concrete adapter from ``pymargins.adapters`` only when you need a
non-default scale that shares a result class with another adapter (e.g.
``LifelinesCoxPHSurvivalAdapter`` for survival probability vs the
auto-detected ``LifelinesCoxPHAdapter`` for hazard ratio), when wrapping a
framework with no native inference (``SklearnBootstrapAdapter``), or when
constructing an adapter explicitly to override auto-detection or pass extra
arguments such as ``training_data=``.

.. currentmodule:: pymargins.adapters

.. automodule:: pymargins.adapters

.. currentmodule:: pymargins

Gradient helpers (for adapter authors)
--------------------------------------

.. autosummary::
    :toctree: _autosummary

    make_predict_with_fd_jvp
    make_glm_jvp_wrapper
    GradientBackend
    InferenceMethod

Transform pipeline (bootstrap)
------------------------------

.. autosummary::
    :toctree: _autosummary

    reimpute
    drop_outliers
    trim

Multiple imputation (pooling)
-----------------------------

Rubin's-rules combinator over ``MarginsResult`` objects from M completed
datasets. The artifacts-side counterpart to the bootstrap ``reimpute`` stage:
use ``pool_imputations`` when you already hold M precomputed imputations, and
``reimpute`` when you have a re-runnable imputer.

.. autosummary::
    :toctree: _autosummary

    pool_imputations
    ImputationDiagnostic

Matching
--------

.. autosummary::
    :toctree: _autosummary
    :recursive:

    PysmatchClient
