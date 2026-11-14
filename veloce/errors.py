"""Veloce compiler diagnostics."""


class VeloceError(Exception):
    """Base class for all Veloce compile-time errors."""

    def __init__(self, message, line=None, col=None):
        self.message = message
        self.line = line
        self.col = col
        super().__init__(self.render())

    def render(self):
        loc = ""
        if self.line is not None:
            loc = f"line {self.line}"
            if self.col is not None:
                loc += f":{self.col}"
            loc += ": "
        return f"{loc}{self.message}"


class LexError(VeloceError):
    pass


class ParseError(VeloceError):
    pass


class TypeError_(VeloceError):
    pass


class SemanticError(VeloceError):
    pass
