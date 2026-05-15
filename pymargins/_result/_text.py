"""Text formatting helpers for result display."""


class SummaryString(str):
    """String subclass whose repr displays raw content in IPython/Jupyter.

    Plain Python strings rendered as the last expression of a notebook cell
    are shown via ``repr()``, which escapes newlines and wraps the text in
    quotes.  ``SummaryString`` overrides ``__repr__`` to return the raw
    string content, so multi-line tables display cleanly without needing
    an explicit ``print(...)`` call.
    """

    def __repr__(self) -> str:
        return str(self)
