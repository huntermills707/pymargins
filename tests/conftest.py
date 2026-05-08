"""Pytest configuration and patsy compatibility patch."""

# Python 3.12's traceback suggestion computation iterates over frame.f_locals.
# When patsy passes a VarLookupDict as the locals dict to eval(), and an
# exception is raised inside that eval(), Python's _compute_suggestion_error
# tries to list(frame.f_locals) which calls __getitem__(0), __getitem__(1), ...
# on the VarLookupDict. Since VarLookupDict doesn't implement __iter__ or keys(),
# this raises KeyError and crashes pytest during report formatting.

try:
    import patsy.eval

    class _IterableVarLookupDict(patsy.eval.VarLookupDict):
        def __iter__(self):
            seen = set()
            for d in self._dicts:
                for k in d:
                    if k not in seen:
                        seen.add(k)
                        yield k

        def keys(self):
            return list(self)

    patsy.eval.VarLookupDict = _IterableVarLookupDict
except Exception:
    pass  # patsy not installed or structure changed
