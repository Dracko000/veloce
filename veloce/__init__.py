"""Veloce — a low-bit, zero-allocation DSL & AOT compiler for LLM inference.

Public API:
    compile_source(src)            -> CompileResult
    compile_file(path)             -> CompileResult
    CompileResult.c_source         generated C11 translation unit
    CompileResult.diagnostics      dict of compiler statistics
    CompileResult.memory           dict: per-function arena plan
"""

from .lexer import tokenize
from .parser import parse
from .typecheck import Checker
from .memplan import plan_module
from .codegen import CodeGen
from .errors import VeloceError, LexError, ParseError, TypeError_, SemanticError
from . import ast as ast_mod

__version__ = "1.0.0"

__all__ = [
    "compile_source", "compile_file", "CompileResult",
    "tokenize", "parse", "Checker", "plan_module", "CodeGen",
    "VeloceError", "LexError", "ParseError", "TypeError_", "SemanticError",
]


class CompileResult:
    def __init__(self, module, c_source, diagnostics, memory):
        self.module = module
        self.c_source = c_source
        self.diagnostics = diagnostics
        self.memory = memory

    def report(self):
        lines = []
        lines.append(f"module '{self.module.name}'  ({len(self.module.functions)} functions)")
        lines.append(f"  trit matmuls (matmul-free): {self.diagnostics['trit_matmul']}")
        lines.append(f"  float matmuls:             {self.diagnostics['fp_matmul']}")
        for fn, plan in self.memory.items():
            r = plan.report()
            lines.append(
                f"  arena[{fn}]: peak {r['peak_bytes']} B "
                f"(naive {r['naive_bytes']} B, reuse saves {r['reuse_savings_pct']}%)")
        return "\n".join(lines)


def resolve_dims(module):
    """Substitute symbolic dimensions (declared with `dim NAME = N`) with
    concrete integers throughout the module's types, so the arena planner can
    size buffers statically (compile-time memory planning, FR-3)."""
    dims = getattr(module, "dims", {}) or {}

    def resolve_dim(d):
        if d.is_symbol:
            if d.value not in dims:
                raise VeloceError(
                    f"dimension '{d.value}' is not declared; add `dim {d.value} = N` "
                    f"at the top of the module")
            return ast_mod.Dim(dims[d.value])
        return d

    def resolve_type(t):
        if isinstance(t, ast_mod.TensorType):
            return ast_mod.TensorType(t.dtype, tuple(resolve_dim(d) for d in t.dims))
        return t

    for fn in module.functions:
        for p in fn.params:
            p.type = resolve_type(p.type)
        fn.ret_types = [resolve_type(t) for t in fn.ret_types]


def compile_source(src, module_name=None):
    tokens = tokenize(src)
    module = parse(tokens)
    if module_name:
        module.name = module_name
    resolve_dims(module)
    checker = Checker(module)
    diag = checker.check()
    plans = plan_module(module)
    gen = CodeGen(module, plans, diag)
    c_source = gen.generate()
    memory = {fn: plan for fn, plan in plans.items()}
    return CompileResult(module, c_source, diag, memory)


def compile_file(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return compile_source(src, module_name=name)
