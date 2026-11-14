"""Abstract syntax tree node definitions for Veloce."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Dim:
    """A tensor dimension: a concrete int or a symbolic name (str)."""

    value: Any

    @property
    def is_symbol(self):
        return isinstance(self.value, str)

    def __str__(self):
        return str(self.value)


@dataclass(frozen=True)
class ScalarType:
    name: str  # trit, fp16, f32, i32, ...

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class TensorType:
    dtype: ScalarType
    dims: Tuple[Dim, ...]

    def __str__(self):
        dims = ", ".join(str(d) for d in self.dims)
        return f"{self.dtype.name}[{dims}]"

    @property
    def rank(self):
        return len(self.dims)

    def numel(self):
        total = 1
        for d in self.dims:
            if d.is_symbol:
                return None
            total *= d.value
        return total


# --------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------
@dataclass
class Expr:
    line: int = 0
    col: int = 0


@dataclass
class IntLit(Expr):
    value: int = 0


@dataclass
class FloatLit(Expr):
    value: float = 0.0


@dataclass
class BoolLit(Expr):
    value: bool = False


@dataclass
class StrLit(Expr):
    value: str = ""


@dataclass
class Var(Expr):
    name: str = ""


@dataclass
class Index(Expr):
    base: Expr = None
    index: Expr = None


@dataclass
class Cast(Expr):
    expr: Expr = None
    dtype: ScalarType = None


@dataclass
class UnaryOp(Expr):
    op: str = ""
    operand: Expr = None


@dataclass
class BinOp(Expr):
    op: str = ""
    left: Expr = None
    right: Expr = None


@dataclass
class Call(Expr):
    func: str = ""
    args: List[Expr] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


@dataclass
class TupleExpr(Expr):
    items: List[Expr] = field(default_factory=list)


# --------------------------------------------------------------------------
# Statements
# --------------------------------------------------------------------------
@dataclass
class Stmt:
    line: int = 0
    col: int = 0


@dataclass
class Param:
    name: str
    type: Any  # ScalarType | TensorType
    rw: bool = False  # caller-owned mutable buffer (e.g. KV cache)


@dataclass
class FnDecl(Stmt):
    name: str = ""
    params: List[Param] = field(default_factory=list)
    ret_types: List[Any] = field(default_factory=list)
    body: List[Stmt] = field(default_factory=list)
    is_extern: bool = False


@dataclass
class LetStmt(Stmt):
    names: List[str] = field(default_factory=list)
    mut: bool = False
    type_ann: Any = None
    expr: Expr = None


@dataclass
class AssignStmt(Stmt):
    targets: List[Expr] = field(default_factory=list)
    expr: Expr = None


@dataclass
class ReturnStmt(Stmt):
    exprs: List[Expr] = field(default_factory=list)


@dataclass
class IfStmt(Stmt):
    cond: Expr = None
    then_body: List[Stmt] = field(default_factory=list)
    else_body: List[Stmt] = field(default_factory=list)


@dataclass
class WhileStmt(Stmt):
    cond: Expr = None
    body: List[Stmt] = field(default_factory=list)


@dataclass
class BreakStmt(Stmt):
    pass


@dataclass
class ContinueStmt(Stmt):
    pass


@dataclass
class ExprStmt(Stmt):
    expr: Expr = None


@dataclass
class Module:
    name: str = ""
    functions: List[FnDecl] = field(default_factory=list)
    dims: dict = field(default_factory=dict)
    target_arch: str = "native"
    cache_line_bytes: int = 64
