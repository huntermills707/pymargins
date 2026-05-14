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

Gradient helpers (for adapter authors)
--------------------------------------

.. autosummary::
    :toctree: _autosummary

    make_predict_with_fd_jvp
    make_glm_jvp_wrapper
    GradientBackend
    InferenceMethod

Matching
--------

.. autosummary::
    :toctree: _autosummary
    :recursive:

    PysmatchClient
