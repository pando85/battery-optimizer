"""Climate optimizer, offline part (C2): configuration, data loading, thermal models and response.

Read-only toolkit on top of the climate feature set (``build-features --signal-set climate``) and
a canonical dataset's prices.  It never writes into a dataset or feature-set directory and has no
runtime or actuation surface.  See ``docs/planning/climate-optimizer-implementation.md``.
"""
