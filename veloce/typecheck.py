"""Semantic analysis: symbol resolution, type & shape checking.

Annotates the AST in place with resolved types and unique C identifiers
(``cname``) so that code generation is a straightforward tree walk.  All
intermediate activations are f32; trit / fp16 / i4 are *storage* types used
for weights and model buffers, matching the PRD's "quantize storage, compute
in float" model.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from . import ast
from .errors import TypeError_, SemanticError

FLOAT_PRIM = "f32"


def is_float(t):
    return isinstance(t, ast.ScalarType) and t.name in ("f32", "fp32", "fp16")


def is_int(t):
    return isinstance(t, ast.ScalarType) and t.name in ("i32", "i64", "i8", "u8", "i2", "i4")


@dataclass
class Symbol:
    name: str
    type: Any
    cname: str
    mut: bool = False
    is_param: bool = False
    # For tensors bound to weights, remember the storage dtype.
    storage_dtype: str = None


@dataclass
class FuncSig:
    name: str
    params: List[Any]
    rets: List[Any]
    is_extern: bool = False
    decl: Any = None


@dataclass
class TupleType:
    items: List[Any]


class Checker:
    def __init__(self, module):
        self.module = module
        self.sigs = {}
        self.counter = 0
        self.diag = {
            "kernels": [],          # fused/computed kernels discovered
            "trit_matmul": 0,
            "fp_matmul": 0,
            "trit_bytes": 0,
            "f32_bytes": 0,
        }

    def new_cname(self, base="v"):
        self.counter += 1
        return f"{base}{self.counter}"

    # -- module -----------------------------------------------------------
    def check(self):
        # Pass 1: collect signatures.
        for fn in self.module.functions:
            params = [p.type for p in fn.params]
            self.sigs[fn.name] = FuncSig(
                name=fn.name, params=params, rets=list(fn.ret_types),
                is_extern=fn.is_extern, decl=fn,
            )
        # Assign C names to params.
        for fn in self.module.functions:
            for p in fn.params:
                p.cname = self.new_cname("p_" + p.name)
            fn.ret_types = list(fn.ret_types)
        # Pass 2: check bodies.
        for fn in self.module.functions:
            if fn.is_extern:
                continue
            self.check_fn(fn)
        return self.diag

    def check_fn(self, fn):
        sig = self.sigs[fn.name]
        scope = {}
        for p, ptype in zip(fn.params, sig.params):
            self.validate_type(ptype, p)
            scope[p.name] = Symbol(p.name, ptype, p.cname, mut=bool(getattr(p, "rw", False)),
                                   is_param=True,
                                   storage_dtype=self._storage_of(ptype))
        self.scope_stack = [scope]
        self.fn = fn
        self.ret_types = sig.rets
        self.saw_return = False
        for stmt in fn.body:
            self.check_stmt(stmt)
        # A function with declared returns must return on all paths; we do a
        # light check (at least one return) rather than full CFG analysis.
        if sig.rets and not self.saw_return:
            raise SemanticError(
                f"function '{fn.name}' declares a return type but never returns",
                fn.line, fn.col)

    def _storage_of(self, t):
        if isinstance(t, ast.TensorType):
            return t.dtype.name
        return t.name if isinstance(t, ast.ScalarType) else None

    def validate_type(self, t, node):
        if isinstance(t, ast.TensorType):
            if t.rank == 0:
                raise SemanticError("zero-rank tensors are not supported", getattr(node, "line", 0),
                                    getattr(node, "col", 0))
            if t.rank > 2:
                raise SemanticError("tensors of rank > 2 are not supported in v1.0",
                                    getattr(node, "line", 0), getattr(node, "col", 0))

    # -- scopes -----------------------------------------------------------
    def lookup(self, name, node):
        for scope in reversed(self.scope_stack):
            if name in scope:
                return scope[name]
        raise SemanticError(f"undefined variable '{name}'", node.line, node.col)

    def declare(self, name, sym, node):
        scope = self.scope_stack[-1]
        if name in scope:
            raise SemanticError(f"variable '{name}' is already defined", node.line, node.col)
        scope[name] = sym

    # -- statements -------------------------------------------------------
    def check_stmt(self, stmt):
        if isinstance(stmt, ast.LetStmt):
            self.check_let(stmt)
        elif isinstance(stmt, ast.AssignStmt):
            self.check_assign(stmt)
        elif isinstance(stmt, ast.ReturnStmt):
            self.check_return(stmt)
        elif isinstance(stmt, ast.IfStmt):
            self.check_if(stmt)
        elif isinstance(stmt, ast.WhileStmt):
            self.check_while(stmt)
        elif isinstance(stmt, (ast.BreakStmt, ast.ContinueStmt)):
            pass
        elif isinstance(stmt, ast.ExprStmt):
            t = self.check_expr(stmt.expr)
            stmt.ctype = t
        else:
            raise SemanticError(f"unsupported statement {type(stmt).__name__}",
                                getattr(stmt, "line", 0), getattr(stmt, "col", 0))

    def check_let(self, stmt):
        t = self.check_expr(stmt.expr)
        # Multiple binding:  let a, b = fn()
        if len(stmt.names) > 1:
            if not isinstance(t, TupleType):
                raise SemanticError(
                    "multiple names on the left require a call returning multiple values",
                    stmt.line, stmt.col)
            if len(t.items) != len(stmt.names):
                raise SemanticError(
                    f"expected {len(t.items)} values but got {len(stmt.names)} names",
                    stmt.line, stmt.col)
            types = t.items
        else:
            types = [t]
        stmt.symbols = []
        for name, vt in zip(stmt.names, types):
            self.validate_type(vt, stmt)
            sym = Symbol(name, vt, self.new_cname(name), mut=stmt.mut,
                         storage_dtype=self._storage_of(vt))
            self.declare(name, sym, stmt)
            stmt.symbols.append(sym)

    def check_assign(self, stmt):
        rhs_t = self.check_expr(stmt.expr)
        if len(stmt.targets) > 1:
            if not isinstance(rhs_t, TupleType) or len(rhs_t.items) != len(stmt.targets):
                raise SemanticError("multiple assignment target/value mismatch",
                                    stmt.line, stmt.col)
            types = rhs_t.items
        else:
            types = [rhs_t]
        for tgt, vt in zip(stmt.targets, types):
            if isinstance(tgt, ast.Var):
                sym = self.lookup(tgt.name, tgt)
                if not sym.mut:
                    raise SemanticError(
                        f"cannot assign to immutable variable '{tgt.name}' (declare with `let`, "
                        f"not `const`)", stmt.line, stmt.col)
                if not self.types_equal(sym.type, vt):
                    raise TypeError_(
                        f"type mismatch in assignment to '{tgt.name}': "
                        f"{sym.type} vs {vt}", stmt.line, stmt.col)
                tgt.symbol = sym
                tgt.ctype = vt
            elif isinstance(tgt, ast.Index):
                base_t = self.check_expr(tgt.base)
                if not isinstance(base_t, ast.TensorType):
                    raise TypeError_("indexed assignment requires a tensor/vector",
                                     tgt.line, tgt.col)
                idx_t = self.check_expr(tgt.index)
                base_sym = tgt.base.symbol if isinstance(tgt.base, ast.Var) else None
                if base_sym and not base_sym.mut:
                    raise SemanticError(
                        f"cannot write into immutable '{tgt.base.name}'", tgt.line, tgt.col)
                # `m[i, j]` writes one element; `m[i]` on a 2-D tensor writes a
                # whole row, so the value being assigned must be a matching
                # 1-D tensor, not a scalar.  Without this the two forms were
                # indistinguishable to codegen, which emitted an element store
                # for a row assignment and cast the RHS pointer to float.
                if isinstance(tgt.index, ast.TupleExpr):
                    if base_t.rank != 2:
                        raise TypeError_("two-index assignment requires a 2-D tensor",
                                         tgt.line, tgt.col)
                    tgt.ctype = base_t.dtype
                elif base_t.rank == 2:
                    row_t = ast.TensorType(base_t.dtype, base_t.dims[1:])
                    # `[1, D]` is the shape every matmul in this language
                    # returns, so writing one into a cache row must work --
                    # otherwise a KV-cache append needs a pointless reshape.
                    squeezed = (isinstance(vt, ast.TensorType) and vt.rank == 2
                                and vt.dims[0].value == 1
                                and self.types_equal(
                                    ast.TensorType(vt.dtype, vt.dims[1:]), row_t))
                    if not (self.types_equal(row_t, vt) or squeezed):
                        raise TypeError_(
                            f"row assignment to '{getattr(tgt.base, 'name', '?')}' "
                            f"needs a {row_t} value, got {vt}", stmt.line, stmt.col)
                    tgt.ctype = row_t
                else:
                    if isinstance(vt, ast.TensorType):
                        raise TypeError_(
                            f"cannot assign a tensor into one element of "
                            f"'{getattr(tgt.base, 'name', '?')}'", stmt.line, stmt.col)
                    tgt.ctype = base_t.dtype
                tgt.row_store = base_t.rank == 2 and not isinstance(tgt.index, ast.TupleExpr)
            else:
                raise SemanticError("invalid assignment target", stmt.line, stmt.col)

    def check_return(self, stmt):
        self.saw_return = True
        exprs = stmt.exprs
        if not self.ret_types:
            if exprs:
                raise TypeError_(f"function '{self.fn.name}' returns nothing", stmt.line, stmt.col)
            return
        if len(exprs) != len(self.ret_types):
            raise TypeError_(
                f"function '{self.fn.name}' expects {len(self.ret_types)} return value(s), "
                f"got {len(exprs)}", stmt.line, stmt.col)
        for e, want in zip(exprs, self.ret_types):
            got = self.check_expr(e)
            self.require_assignable(want, got, e)

    def check_if(self, stmt):
        self.check_expr(stmt.cond)
        self.scope_stack.append({})
        for s in stmt.then_body:
            self.check_stmt(s)
        self.scope_stack.pop()
        self.scope_stack.append({})
        for s in stmt.else_body:
            self.check_stmt(s)
        self.scope_stack.pop()

    def check_while(self, stmt):
        self.check_expr(stmt.cond)
        self.scope_stack.append({})
        for s in stmt.body:
            self.check_stmt(s)
        self.scope_stack.pop()

    # -- type helpers -----------------------------------------------------
    def types_equal(self, a, b):
        return str(a) == str(b)

    def require_assignable(self, want, got, node):
        if str(want) == str(got):
            return
        # allow scalar widening i32 -> f32
        if isinstance(want, ast.ScalarType) and isinstance(got, ast.ScalarType):
            if is_float(want) and (is_float(got) or is_int(got)):
                return
        raise TypeError_(f"expected {want} but got {got}", node.line, node.col)

    # -- expressions ------------------------------------------------------
    def check_expr(self, e):
        t = self._check_expr(e)
        e.ctype = t
        return t

    def _check_expr(self, e):
        if isinstance(e, ast.IntLit):
            return ast.ScalarType("i32")
        if isinstance(e, ast.FloatLit):
            return ast.ScalarType("f32")
        if isinstance(e, ast.BoolLit):
            return ast.ScalarType("bool")
        if isinstance(e, ast.StrLit):
            return ast.ScalarType("str")
        if isinstance(e, ast.Var):
            sym = self.lookup(e.name, e)
            e.symbol = sym
            return sym.type
        if isinstance(e, ast.Index):
            bt = self.check_expr(e.base)
            if not isinstance(bt, ast.TensorType):
                raise TypeError_("cannot index a scalar", e.line, e.col)
            if isinstance(e.index, ast.TupleExpr):
                e.index_dims = [self.check_expr(i) for i in e.index.items]
                if bt.rank == 2:
                    e.lower = "index2"
                    return bt.dtype  # scalar element
                raise TypeError_("multi-index only valid on 2-D tensors", e.line, e.col)
            self.check_expr(e.index)
            if bt.rank == 1:
                e.lower = "index1"
                return bt.dtype  # element
            # rank-2 indexing with single index -> row (1D)
            e.lower = "indexrow"
            return ast.TensorType(bt.dtype, bt.dims[1:])
        if isinstance(e, ast.Cast):
            self.check_expr(e.expr)
            return ast.ScalarType(FLOAT_PRIM if e.dtype.name in ("f32", "fp32") else e.dtype.name)
        if isinstance(e, ast.UnaryOp):
            ot = self.check_expr(e.operand)
            if e.op == "!":
                return ast.ScalarType("bool")
            if isinstance(ot, ast.TensorType):
                return ot
            return ot
        if isinstance(e, ast.BinOp):
            return self.check_binop(e)
        if isinstance(e, ast.Call):
            return self.check_call(e)
        if isinstance(e, ast.TupleExpr):
            return TupleType([self.check_expr(i) for i in e.items])
        raise SemanticError(f"unsupported expression {type(e).__name__}",
                            getattr(e, "line", 0), getattr(e, "col", 0))

    def check_binop(self, e):
        lt = self.check_expr(e.left)
        rt = self.check_expr(e.right)
        op = e.op
        if op in ("==", "!=", "<", ">", "<=", ">=", "&&", "||"):
            return ast.ScalarType("bool")
        if op == "@":
            return self.check_matmul(e, lt, rt)
        # elementwise
        if isinstance(lt, ast.TensorType) and isinstance(rt, ast.TensorType):
            # bias broadcast: [R,C] (op) [C]  ->  [R,C]
            if lt.rank == 2 and rt.rank == 1:
                if not (rt.dims[0].is_symbol or lt.dims[1].is_symbol
                        or rt.dims[0].value == lt.dims[1].value):
                    raise TypeError_(
                        f"broadcast mismatch: {lt} vs {rt}", e.line, e.col)
                e.bcast = ("right", lt.dims[1].value)
                return ast.TensorType(ast.ScalarType(FLOAT_PRIM), lt.dims)
            if lt.rank == 1 and rt.rank == 2:
                if not (lt.dims[0].is_symbol or rt.dims[1].is_symbol
                        or lt.dims[0].value == rt.dims[1].value):
                    raise TypeError_(
                        f"broadcast mismatch: {lt} vs {rt}", e.line, e.col)
                e.bcast = ("left", rt.dims[1].value)
                return ast.TensorType(ast.ScalarType(FLOAT_PRIM), rt.dims)
            out = self.unify_shapes(lt, rt, e)
            return ast.TensorType(ast.ScalarType(FLOAT_PRIM), out)
        if isinstance(lt, ast.TensorType) and isinstance(rt, ast.ScalarType):
            return ast.TensorType(ast.ScalarType(FLOAT_PRIM), lt.dims)
        if isinstance(lt, ast.ScalarType) and isinstance(rt, ast.TensorType):
            return ast.TensorType(ast.ScalarType(FLOAT_PRIM), rt.dims)
        # scalar
        if is_float(lt) or is_float(rt) or op == "/":
            return ast.ScalarType(FLOAT_PRIM)
        return ast.ScalarType("i32")

    def unify_shapes(self, lt, rt, node):
        if lt.rank != rt.rank:
            raise TypeError_(
                f"shape mismatch: {lt} vs {rt} (different rank)", node.line, node.col)
        out = []
        for a, b in zip(lt.dims, rt.dims):
            if a.is_symbol or b.is_symbol:
                out.append(a if a.is_symbol else b)
            elif a.value == b.value:
                out.append(a)
            else:
                raise TypeError_(
                    f"shape mismatch: dimension {a} vs {b} in {lt} vs {rt}",
                    node.line, node.col)
        return tuple(out)

    def check_matmul(self, e, lt, rt):
        if not (isinstance(lt, ast.TensorType) and isinstance(rt, ast.TensorType)):
            raise TypeError_("matmul '@' requires two tensors", e.line, e.col)
        if lt.rank != 2 or rt.rank != 2:
            raise TypeError_("matmul '@' requires 2-D tensors", e.line, e.col)
        if rt.dtype.name == "trit":
            raise TypeError_(
                "trit weights must go through `matmul_trit(x, w)` with w shaped [out, in] "
                "so it can lower to add/sub", e.line, e.col)
        if rt.dtype.name == "i4":
            raise TypeError_(
                "i4 weights must go through `matmul_i4(x, w)` with w shaped [out, in]",
                e.line, e.col)
        k1, n = lt.dims[1], rt.dims[0]
        if not (k1.is_symbol or n.is_symbol or k1.value == n.value):
            raise TypeError_(
                f"matmul inner dimension mismatch: {lt} @ {rt}", e.line, e.col)
        out = (lt.dims[0], rt.dims[1])
        e.lower = "matmul_f32"
        self.diag["fp_matmul"] += 1
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), out)

    def check_call(self, e):
        name = e.func
        # user-defined function
        if name in self.sigs:
            sig = self.sigs[name]
            if len(e.args) != len(sig.params):
                raise TypeError_(
                    f"'{name}' expects {len(sig.params)} argument(s), got {len(e.args)}",
                    e.line, e.col)
            for a, want in zip(e.args, sig.params):
                got = self.check_expr(a)
                if not self.types_equal(want, got):
                    # allow scalar widening
                    if not (isinstance(want, ast.ScalarType) and isinstance(got, ast.ScalarType)
                            and is_float(want)):
                        raise TypeError_(
                            f"argument to '{name}': expected {want} but got {got}",
                            a.line, a.col)
            if len(sig.rets) == 1:
                return sig.rets[0]
            return TupleType(list(sig.rets))

        # builtins
        handler = getattr(self, f"bi_{name}", None)
        if handler is None:
            raise SemanticError(f"unknown function '{name}'", e.line, e.col)
        return handler(e)

    # -- builtins ---------------------------------------------------------
    def _tensor_arg(self, e, idx, fname, node):
        if idx >= len(e.args):
            raise TypeError_(f"'{fname}' missing argument {idx + 1}", node.line, node.col)
        t = self.check_expr(e.args[idx])
        if not isinstance(t, ast.TensorType):
            raise TypeError_(f"'{fname}' argument {idx + 1} must be a tensor, got {t}",
                             e.args[idx].line, e.args[idx].col)
        return t

    def bi_matmul(self, e):
        lt = self._tensor_arg(e, 0, "matmul", e)
        rt = self._tensor_arg(e, 1, "matmul", e)
        return self.check_matmul(e, lt, rt)

    def bi_matmul_trit(self, e):
        lt = self._tensor_arg(e, 0, "matmul_trit", e)
        rt = self._tensor_arg(e, 1, "matmul_trit", e)
        if rt.dtype.name != "trit":
            raise TypeError_("matmul_trit requires a trit (1.58-bit) weight tensor",
                             e.line, e.col)
        if lt.rank != 2 or rt.rank != 2:
            raise TypeError_("matmul_trit requires 2-D tensors", e.line, e.col)
        n, k = lt.dims[1], rt.dims[1]
        if not (n.is_symbol or k.is_symbol or n.value == k.value):
            raise TypeError_(
                f"matmul_trit inner dimension mismatch: x{list(lt.dims)} vs "
                f"w{list(rt.dims)} (weights must be [out, in])", e.line, e.col)
        e.lower = "matmul_trit"
        self.diag["trit_matmul"] += 1
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM),
                              (lt.dims[0], rt.dims[0]))

    def bi_matmul_i4(self, e):
        """matmul_i4(x: f32[M,K], w: i4[N,K]) -> f32[M,N].

        Packed 4-bit weights: two signed nibbles per byte, values in [-7, 7].
        The runtime decodes with a LUT and accumulates in f32.
        """
        lt = self._tensor_arg(e, 0, "matmul_i4", e)
        rt = self._tensor_arg(e, 1, "matmul_i4", e)
        if rt.dtype.name != "i4":
            raise TypeError_("matmul_i4 requires an i4 (4-bit) weight tensor",
                             e.line, e.col)
        if lt.rank != 2 or rt.rank != 2:
            raise TypeError_("matmul_i4 requires 2-D tensors", e.line, e.col)
        n, k = lt.dims[1], rt.dims[1]
        if not (n.is_symbol or k.is_symbol or n.value == k.value):
            raise TypeError_(
                f"matmul_i4 inner dimension mismatch: x{list(lt.dims)} vs "
                f"w{list(rt.dims)} (weights must be [out, in])", e.line, e.col)
        e.lower = "matmul_i4"
        self.diag["i4_matmul"] = self.diag.get("i4_matmul", 0) + 1
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM),
                              (lt.dims[0], rt.dims[0]))

    def _lowbit_matmul(self, e, fname, dtype, label):
        """Shared shape check for a [N,K]-weight low-bit matmul.

        x: f32[M,K], w: <dtype>[N,K] -> f32[M,N].  Weights are stored
        [out, in] like every other Veloce weight tensor.
        """
        lt = self._tensor_arg(e, 0, fname, e)
        rt = self._tensor_arg(e, 1, fname, e)
        if rt.dtype.name != dtype:
            raise TypeError_(f"{fname} requires a {dtype} ({label}) weight tensor",
                             e.line, e.col)
        if lt.rank != 2 or rt.rank != 2:
            raise TypeError_(f"{fname} requires 2-D tensors", e.line, e.col)
        n, k = lt.dims[1], rt.dims[1]
        if not (n.is_symbol or k.is_symbol or n.value == k.value):
            raise TypeError_(
                f"{fname} inner dimension mismatch: x{list(lt.dims)} vs "
                f"w{list(rt.dims)} (weights must be [out, in])", e.line, e.col)
        e.lower = fname
        key = f"{dtype}_matmul"
        self.diag[key] = self.diag.get(key, 0) + 1
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), (lt.dims[0], rt.dims[0]))

    def bi_matmul_i2(self, e):
        """matmul_i2(x: f32[M,K], w: i2[N,K]) -> f32[M,N].

        Packed 2-bit weights: four signed values per byte, values in [-2, 1].
        """
        return self._lowbit_matmul(e, "matmul_i2", "i2", "2-bit")

    def bi_matmul_fp8(self, e):
        """matmul_fp8(x: f32[M,K], w: fp8[N,K]) -> f32[M,N].

        E4M3 weights, one byte per value, decoded to f32 during the dot.
        """
        return self._lowbit_matmul(e, "matmul_fp8", "fp8", "E4M3")

    def _lowbit_matmul_scaled(self, e, fname, dtype, label):
        """[N,K] low-bit weights plus one f32 scale per output row."""
        lt = self._tensor_arg(e, 0, fname, e)
        wt = self._tensor_arg(e, 1, fname, e)
        st = self._tensor_arg(e, 2, fname, e)
        if wt.dtype.name != dtype:
            raise TypeError_(f"{fname} requires a {dtype} ({label}) weight tensor",
                             e.line, e.col)
        if st.dtype.name != "f32" or st.rank != 1:
            raise TypeError_(f"{fname} scales must be f32[N]", e.line, e.col)
        if lt.rank != 2 or wt.rank != 2:
            raise TypeError_(f"{fname} requires 2-D tensors", e.line, e.col)
        if not (lt.dims[1].is_symbol or wt.dims[1].is_symbol
                or lt.dims[1].value == wt.dims[1].value):
            raise TypeError_(f"{fname} inner dimension mismatch", e.line, e.col)
        if not (wt.dims[0].is_symbol or st.dims[0].is_symbol
                or wt.dims[0].value == st.dims[0].value):
            raise TypeError_(f"{fname} scales must have one entry per output feature",
                             e.line, e.col)
        e.lower = fname
        key = f"{dtype}_matmul"
        self.diag[key] = self.diag.get(key, 0) + 1
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), (lt.dims[0], wt.dims[0]))

    def bi_matmul_i2_s(self, e):
        return self._lowbit_matmul_scaled(e, "matmul_i2_s", "i2", "2-bit")

    def bi_matmul_fp8_s(self, e):
        return self._lowbit_matmul_scaled(e, "matmul_fp8_s", "fp8", "E4M3")

    def bi_zeros_i32(self, e):
        dims = []
        for a in e.args:
            self.check_expr(a)
            if isinstance(a, ast.IntLit):
                dims.append(ast.Dim(int(a.value)))
            else:
                dims.append(ast.Dim(0))
        e.lower = "zeros"
        if len(dims) == 1:
            return ast.TensorType(ast.ScalarType("i32"), (dims[0],))
        if len(dims) == 2:
            return ast.TensorType(ast.ScalarType("i32"), (dims[0], dims[1]))
        raise TypeError_("zeros_i32 takes 1 or 2 dimensions", e.line, e.col)

    def _group_scaled_matmul(self, e, fname, dtype, label):
        """matmul_Xg(x: f32[M,K], w: X[N,K], scales: f32[N,G]) -> f32[M,N].

        Shared by `matmul_i4g`, `matmul_i2g` and `matmul_fp8g`: the only
        difference between the three is the weight dtype and the kernel the
        backend lowers to.
        """
        lt = self._tensor_arg(e, 0, fname, e)
        wt = self._tensor_arg(e, 1, fname, e)
        st = self._tensor_arg(e, 2, fname, e)
        if wt.dtype.name != dtype:
            raise TypeError_(f"{fname} requires a {dtype} ({label}) weight tensor",
                             e.line, e.col)
        if st.dtype.name != "f32" or st.rank != 2:
            raise TypeError_(f"{fname} scales must be f32[N, groups]", e.line, e.col)
        if lt.rank != 2 or wt.rank != 2:
            raise TypeError_(f"{fname} requires 2-D tensors", e.line, e.col)
        if not (lt.dims[1].is_symbol or wt.dims[1].is_symbol
                or lt.dims[1].value == wt.dims[1].value):
            raise TypeError_(f"{fname} inner dimension mismatch", e.line, e.col)
        if not (wt.dims[0].is_symbol or st.dims[0].is_symbol
                or wt.dims[0].value == st.dims[0].value):
            raise TypeError_(f"{fname} scales rows must match output features",
                             e.line, e.col)
        # Group count must divide K exactly: the runtime is handed a group
        # *size* and derives the count back as K / G, so a ragged shape
        # (K=100 over 7 groups) would silently scale rows against the wrong
        # slices.  Symbolic dims can't be checked here; they are resolved by
        # the time codegen runs.
        k, ng = wt.dims[1], st.dims[1]
        if not (k.is_symbol or ng.is_symbol):
            if ng.value <= 0 or k.value % ng.value != 0:
                raise TypeError_(
                    f"{fname}: scales declare {ng.value} group(s) but the "
                    f"inner dimension is K={k.value}; the group count must "
                    f"divide K exactly", e.line, e.col)
        e.lower = fname
        self.diag[f"{dtype}_matmul"] = self.diag.get(f"{dtype}_matmul", 0) + 1
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), (lt.dims[0], wt.dims[0]))

    def bi_matmul_i4g(self, e):
        return self._group_scaled_matmul(e, "matmul_i4g", "i4", "4-bit")

    def bi_matmul_i2g(self, e):
        return self._group_scaled_matmul(e, "matmul_i2g", "i2", "2-bit")

    def bi_matmul_fp8g(self, e):
        return self._group_scaled_matmul(e, "matmul_fp8g", "fp8", "E4M3")

    def bi_matmul_trit_s(self, e):
        """matmul_trit_s(x, w: trit[N,K], scales: f32[N]) -> f32[M,N]."""
        lt = self._tensor_arg(e, 0, "matmul_trit_s", e)
        wt = self._tensor_arg(e, 1, "matmul_trit_s", e)
        st = self._tensor_arg(e, 2, "matmul_trit_s", e)
        if wt.dtype.name != "trit":
            raise TypeError_("matmul_trit_s requires a trit weight tensor", e.line, e.col)
        if st.dtype.name != "f32" or st.rank != 1:
            raise TypeError_("matmul_trit_s scales must be f32[N]", e.line, e.col)
        if lt.rank != 2 or wt.rank != 2:
            raise TypeError_("matmul_trit_s requires 2-D tensors", e.line, e.col)
        e.lower = "matmul_trit_s"
        self.diag["trit_matmul"] += 1
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), (lt.dims[0], wt.dims[0]))

    def bi_matmul_f16wt(self, e):
        """matmul_f16wt(x: f32[M,K], w: fp16[N,K]) -> f32[M,N] (HF layout)."""
        lt = self._tensor_arg(e, 0, "matmul_f16wt", e)
        wt = self._tensor_arg(e, 1, "matmul_f16wt", e)
        if wt.dtype.name != "fp16":
            raise TypeError_("matmul_f16wt requires an fp16 weight tensor", e.line, e.col)
        if lt.rank != 2 or wt.rank != 2:
            raise TypeError_("matmul_f16wt requires 2-D tensors", e.line, e.col)
        if not (lt.dims[1].is_symbol or wt.dims[1].is_symbol
                or lt.dims[1].value == wt.dims[1].value):
            raise TypeError_("matmul_f16wt inner dimension mismatch", e.line, e.col)
        e.lower = "matmul_f16wt"
        self.diag["fp_matmul"] += 1
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), (lt.dims[0], wt.dims[0]))

    def bi_rope_heads(self, e):
        """rope_heads(x: f32[T,D], pos: i32, n_heads: int, theta: float)."""
        x = self._tensor_arg(e, 0, "rope_heads", e)
        if len(e.args) < 4:
            raise TypeError_("rope_heads(x, pos, n_heads, theta) needs 4 arguments",
                             e.line, e.col)
        self.check_expr(e.args[1])
        if not isinstance(e.args[2], ast.IntLit) or not isinstance(e.args[3], ast.FloatLit):
            raise TypeError_("rope_heads n_heads/theta must be literals", e.line, e.col)
        if x.rank != 2:
            raise TypeError_("rope_heads requires a 2-D tensor", e.line, e.col)
        e.lower = "rope_heads"
        e.n_heads = int(e.args[2].value)
        e.theta = float(e.args[3].value)
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), x.dims)

    def bi_rmsnorm(self, e):
        x = self._tensor_arg(e, 0, "rmsnorm", e)
        w = self._tensor_arg(e, 1, "rmsnorm", e)
        e.lower = "rmsnorm"
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), x.dims)

    def bi_silu(self, e):
        x = self._tensor_arg(e, 0, "silu", e)
        e.lower = "silu"
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), x.dims)

    def bi_relu(self, e):
        x = self._tensor_arg(e, 0, "relu", e)
        e.lower = "relu"
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), x.dims)

    def bi_softmax(self, e):
        """softmax(x) or softmax(x, mask).

        The two-argument form hides entries instead of biasing them: an entry
        whose mask value is 0 gets probability exactly 0, and the row's max
        and normaliser are computed over visible entries only.  A hidden entry
        can therefore hold any value (a stale cache row, a garbage slot)
        without affecting the result, which the old `score + -1e30` trick
        could not guarantee.
        """
        x = self._tensor_arg(e, 0, "softmax", e)
        if len(e.args) > 1:
            m = self._tensor_arg(e, 1, "softmax", e)
            if len(e.args) > 2:
                raise TypeError_("softmax takes at most two arguments", e.line, e.col)
            if x.rank != 2:
                raise TypeError_("masked softmax requires a 2-D scores tensor",
                                 e.line, e.col)
            if m.dtype.name not in ("i32", "i64", "i8", "u8", "i2", "i4"):
                raise TypeError_(
                    f"softmax mask must be an integer tensor (0 = hidden), got {m.dtype}",
                    e.line, e.col)
            hidden_dim = x.dims[-1]
            if m.rank == 1:
                if not (m.dims[0].is_symbol or hidden_dim.is_symbol
                        or m.dims[0].value == hidden_dim.value):
                    raise TypeError_(
                        f"softmax mask length {m.dims[0]} does not match the "
                        f"last dimension {hidden_dim}", e.line, e.col)
            elif m.rank == 2:
                if not (m.dims[1].is_symbol or hidden_dim.is_symbol
                        or m.dims[1].value == hidden_dim.value):
                    raise TypeError_(
                        f"softmax mask width {m.dims[1]} does not match the "
                        f"last dimension {hidden_dim}", e.line, e.col)
            else:
                raise TypeError_("softmax mask must be 1-D or 2-D", e.line, e.col)
            e.lower = "softmax_mask"
        else:
            e.lower = "softmax"
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), x.dims)

    def bi_transpose(self, e):
        x = self._tensor_arg(e, 0, "transpose", e)
        if x.rank != 2:
            raise TypeError_("transpose requires a 2-D tensor", e.line, e.col)
        e.lower = "transpose"
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), (x.dims[1], x.dims[0]))

    def bi_embedding_seq(self, e):
        table = self._tensor_arg(e, 0, "embedding_seq", e)
        if len(e.args) < 2:
            raise TypeError_("embedding_seq(table, ids) needs an id vector", e.line, e.col)
        it = self.check_expr(e.args[1])
        if not (isinstance(it, ast.TensorType) and it.dtype.name in ("i32", "i64")):
            raise TypeError_("embedding_seq ids must be an i32[...] tensor", e.line, e.col)
        if table.rank != 2:
            raise TypeError_("embedding table must be 2-D", e.line, e.col)
        if table.dtype.name not in ("f32", "fp16"):
            raise TypeError_("embedding_seq table must be f32 or fp16", e.line, e.col)
        e.lower = "embedding_seq"
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM),
                              (it.dims[0], table.dims[1]))

    def bi_meanpool(self, e):
        x = self._tensor_arg(e, 0, "meanpool", e)
        if x.rank != 2:
            raise TypeError_("meanpool requires a 2-D [T, D] tensor", e.line, e.col)
        e.lower = "meanpool"
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), (ast.Dim(1), x.dims[1]))

    def bi_rope(self, e):
        x = self._tensor_arg(e, 0, "rope", e)
        e.lower = "rope"
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), x.dims)

    def bi_embedding(self, e):
        table = self._tensor_arg(e, 0, "embedding", e)
        it = self.check_expr(e.args[1]) if len(e.args) > 1 else None
        if it is None or not isinstance(it, ast.ScalarType):
            raise TypeError_("embedding index must be a scalar integer", e.line, e.col)
        if table.rank != 2:
            raise TypeError_("embedding table must be 2-D", e.line, e.col)
        e.lower = "embedding"
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), (ast.Dim(1), table.dims[1]))

    def bi_zeros(self, e):
        # zeros(n) or zeros(rows, cols) -- dimensions must be integer literals
        dims = []
        for a in e.args:
            t = self.check_expr(a)
            if isinstance(a, ast.IntLit):
                dims.append(ast.Dim(int(a.value)))
            else:
                dims.append(ast.Dim(0))
        e.lower = "zeros"
        if len(dims) == 1:
            return ast.TensorType(ast.ScalarType(FLOAT_PRIM), (dims[0],))
        if len(dims) == 2:
            return ast.TensorType(ast.ScalarType(FLOAT_PRIM), (dims[0], dims[1]))
        raise TypeError_("zeros takes 1 or 2 dimensions", e.line, e.col)

    def bi_exp(self, e):
        x = self.check_expr(e.args[0])
        e.lower = "exp"
        return ast.TensorType(ast.ScalarType(FLOAT_PRIM), x.dims) \
            if isinstance(x, ast.TensorType) else ast.ScalarType(FLOAT_PRIM)

    def bi_sqrt(self, e):
        x = self.check_expr(e.args[0])
        e.lower = "sqrt"
        return x

    def bi_argmax(self, e):
        x = self._tensor_arg(e, 0, "argmax", e)
        e.lower = "argmax"
        return ast.ScalarType("i32")
