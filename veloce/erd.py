"""Introspection of the Veloce compiler's internal data structures.

Builds the entity graph described by ERD.md (MODULE, FUNCTION, MEMORY_ARENA,
MEMORY_SLICE, IR_NODE, FUSED_KERNEL, TENSOR_DESCRIPTOR, DATA_TYPE,
PACKING_RULE, WEIGHT_ASSET, PACKED_BLOCK) from a compiled module, and renders
it as JSON or a Mermaid `erDiagram`.  This lets `velocec erd` show the real
compiler IR/planner state for a given program rather than a static picture.
"""
import uuid

from . import ast
from .typecheck import TupleType

NS = uuid.UUID("6f6c6576-6f63-4500-8000-000000000000")


def _uid(*parts):
    return str(uuid.uuid5(NS, "/".join(str(p) for p in parts)))


OP_CODES = {
    "matmul_trit": "MATMUL_TRIT",
    "matmul_f32": "MATMUL_F32",
    "rmsnorm": "RMS_NORM",
    "silu": "SILU",
    "relu": "RELU",
    "softmax": "SOFTMAX",
    "rope": "ROPE",
    "embedding": "EMBEDDING",
    "transpose": "TRANSPOSE",
    "zeros": "ZEROS",
}


def _op_of(e):
    if isinstance(e, ast.Call):
        lower = getattr(e, "lower", None)
        if lower:
            return OP_CODES.get(lower, lower.upper())
        return e.func.upper()
    if isinstance(e, ast.BinOp):
        if getattr(e, "bcast", None):
            return "BROADCAST_" + {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}[e.op]
        return {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV", "@": "MATMUL"}.get(e.op, e.op)
    if isinstance(e, ast.Index):
        return "INDEX"
    return type(e).__name__.upper()


def _walk_ops(e, out, order):
    """Collect (order, op_code, node_expr) for tensor ops in an expression."""
    if e is None:
        return
    if isinstance(e, ast.Call):
        for a in e.args:
            _walk_ops(a, out, order)
        if isinstance(getattr(e, "ctype", None), ast.TensorType) or getattr(e, "lower", None):
            out.append((order, _op_of(e), e))
    elif isinstance(e, ast.BinOp):
        _walk_ops(e.left, out, order)
        _walk_ops(e.right, out, order)
        if isinstance(getattr(e, "ctype", None), ast.TensorType):
            out.append((order, _op_of(e), e))
    elif isinstance(e, ast.UnaryOp):
        _walk_ops(e.operand, out, order)
    elif isinstance(e, ast.Cast):
        _walk_ops(e.expr, out, order)
    elif isinstance(e, ast.Index):
        _walk_ops(e.base, out, order)
    elif isinstance(e, ast.TupleExpr):
        for i in e.items:
            _walk_ops(i, out, order)


def _iter_tensor_ops(body):
    ops = []
    order = 0
    def visit(stmts):
        nonlocal order
        for s in stmts:
            order += 1
            if isinstance(s, ast.LetStmt):
                _walk_ops(s.expr, ops, order)
            elif isinstance(s, ast.AssignStmt):
                _walk_ops(s.expr, ops, order)
            elif isinstance(s, ast.ReturnStmt):
                for e in s.exprs:
                    _walk_ops(e, ops, order)
            elif isinstance(s, ast.ExprStmt):
                _walk_ops(s.expr, ops, order)
            elif isinstance(s, ast.IfStmt):
                _walk_ops(s.cond, ops, order)
                visit(s.then_body); visit(s.else_body)
            elif isinstance(s, ast.WhileStmt):
                visit(s.body)
    visit(body)
    return ops


def _is_pure_ew(e):
    if not isinstance(getattr(e, "ctype", None), ast.TensorType):
        return False
    if isinstance(e, ast.Var):
        return True
    if isinstance(e, ast.Index):
        return getattr(e, "lower", None) == "indexrow"
    if isinstance(e, ast.Call):
        return isinstance(getattr(e, "ctype", None), ast.TensorType)
    if isinstance(e, ast.Cast):
        return _is_pure_ew(e.expr)
    if isinstance(e, ast.UnaryOp):
        return e.op == "-" and _is_pure_ew(e.operand)
    if isinstance(e, ast.BinOp):
        if e.op not in ("+", "-", "*", "/") or getattr(e, "bcast", None):
            return False
        lt = isinstance(getattr(e.left, "ctype", None), ast.TensorType)
        rt = isinstance(getattr(e.right, "ctype", None), ast.TensorType)
        if lt and rt:
            return _is_pure_ew(e.left) and _is_pure_ew(e.right)
        return _is_pure_ew(e.left if lt else e.right)
    return False


def _dtype_records(module):
    used = set()
    def collect(t):
        if isinstance(t, ast.TensorType):
            used.add(t.dtype.name)
        elif isinstance(t, ast.ScalarType):
            used.add(t.name)
    for fn in module.functions:
        for p in fn.params:
            collect(p.type)
        for r in fn.ret_types:
            collect(r)
        for _, _, e in _iter_tensor_ops(fn.body):
            collect(getattr(e, "ctype", None))
    BIT = {"trit": 1.58, "i2": 2.0, "i4": 4.0, "fp8": 8.0, "fp16": 16.0,
           "f32": 32.0, "fp32": 32.0, "i32": 32.0, "i64": 64.0, "i8": 8.0,
           "u8": 8.0, "bool": 8.0}
    PACK = {"trit": (5, 8, "ADD_SUB_MASK"), "i4": (2, 8, "LUT_UNPACK"),
            "i2": (4, 8, "LUT_UNPACK"), "fp16": (1, 16, "HALF_DECODE")}
    dtypes, packing = [], []
    for name in sorted(used):
        if name in ("bool", "str", "i32", "i64") and name not in BIT:
            continue
        did = _uid("dtype", name)
        requires = name in ("trit", "i2", "i4", "fp8")
        dtypes.append({"dtype_id": did, "name": name,
                       "bit_width": BIT.get(name, 32.0),
                       "requires_bit_packing": requires})
        if name in PACK:
            epw, cbits, instr = PACK[name]
            packing.append({"rule_id": _uid("pack", name), "dtype_id": did,
                            "elements_per_word": epw, "container_bits": cbits,
                            "decode_instruction": instr})
    return dtypes, packing


def build_erd(module, plans, diag):
    mid = _uid("module", module.name)
    ent = {"modules": [], "functions": [], "arenas": [], "slices": [],
           "ir_nodes": [], "fused_kernels": [], "tensors": [],
           "dtypes": [], "packing_rules": [], "weight_assets": [], "packed_blocks": []}
    dtypes, packing = _dtype_records(module)
    ent["dtypes"] = dtypes
    ent["packing_rules"] = packing

    ent["modules"].append({"module_id": mid, "name": module.name,
                           "target_arch": module.target_arch,
                           "cache_line_bytes": module.cache_line_bytes})

    for fn in module.functions:
        if fn.is_extern:
            continue
        fid = _uid("fn", module.name, fn.name)
        aid = _uid("arena", module.name, fn.name)
        plan = plans.get(fn.name)
        # purity: a function is pure if it makes no writes to params/globals —
        # approximated as: has no AssignStmt targeting a param.
        param_names = {p.name for p in fn.params}
        is_pure = True
        def scan(stmts):
            nonlocal is_pure
            for s in stmts:
                if isinstance(s, ast.AssignStmt):
                    for t in s.targets:
                        base = t.base.name if isinstance(t, ast.Index) else getattr(t, "name", None)
                        if base in param_names:
                            is_pure = False
                elif isinstance(s, ast.IfStmt):
                    scan(s.then_body); scan(s.else_body)
                elif isinstance(s, ast.WhileStmt):
                    scan(s.body)
        scan(fn.body)
        ent["functions"].append({"function_id": fid, "module_id": mid,
                                 "arena_id": aid, "identifier": fn.name,
                                 "is_pure": is_pure})
        peak = plan.peak_words * 4 if plan else 0
        naive = plan.naive_words * 4 if plan else 0
        ent["arenas"].append({"arena_id": aid, "module_id": mid, "name": f"{fn.name}.arena",
                              "total_capacity_bytes": peak, "peak_used_bytes": peak,
                              "naive_bytes_no_reuse": naive})
        # slices from allocations
        if plan:
            for a in plan.allocations:
                sid = _uid("slice", fid, id(a))
                ent["slices"].append({
                    "slice_id": sid, "arena_id": aid,
                    "byte_offset": a.offset_words * 4,
                    "byte_length": a.size_words * 4,
                    "birth_step": a.birth, "death_step": a.death})
                t = getattr(a.node, "ctype", None)
                if isinstance(t, ast.TensorType):
                    ent["tensors"].append({
                        "tensor_id": _uid("tensor", sid), "slice_id": sid,
                        "dtype": t.dtype.name,
                        "shape": [d.value for d in t.dims],
                        "is_constant_weight": False})
        # IR nodes
        ops = _iter_tensor_ops(fn.body)
        for order, op, e in ops:
            nid = _uid("node", fid, order, op, id(e))
            ent["ir_nodes"].append({"node_id": nid, "function_id": fid,
                                    "op_code": op, "topological_order": order})
        # fused kernels: each top-level pure-elementwise chain is one kernel
        for order, op, e in ops:
            if _is_pure_ew(e) and (order, id(e)) == (order, id(e)):
                # only when it's a top-level op (avoid double count): detect by
                # whether its parent was also pure-ew -> approximate: count
                # kernels where e is a BinOp whose own node is pure
                if isinstance(e, ast.BinOp):
                    kid = _uid("kernel", fid, order, id(e))
                    ent["fused_kernels"].append({
                        "kernel_id": kid, "symbol_name": f"fused_{fn.name}_{order}",
                        "simd_strategy": "AUTOVEC", "register_tile_m": 1,
                        "register_tile_n": 8})
        # weight assets: tensor params with storage dtype trit/fp16
        for p in fn.params:
            if isinstance(p.type, ast.TensorType) and p.type.dtype.name in ("trit", "fp16"):
                wid = _uid("asset", fid, p.name)
                n = p.type.numel() or 0
                bits = 1.58 if p.type.dtype.name == "trit" else 16.0
                nbytes = ((n + 4) // 5) if p.type.dtype.name == "trit" else n * 2
                ent["weight_assets"].append({
                    "asset_id": wid, "module_id": mid,
                    "file_path": f"{fn.name}.{p.name}.veloce",
                    "file_size_bytes": nbytes, "zero_copy_mmap": True})
                ent["packed_blocks"].append({
                    "block_id": _uid("block", wid), "asset_id": wid,
                    "file_offset": 0, "aligned_size_bytes": nbytes,
                    "tensor_symbol": p.name})

    ent["summary"] = {
        "functions": len(ent["functions"]),
        "ir_nodes": len(ent["ir_nodes"]),
        "fused_kernels": len(ent["fused_kernels"]),
        "memory_slices": len(ent["slices"]),
        "peak_arena_bytes": sum(a["total_capacity_bytes"] for a in ent["arenas"]),
        "weight_assets": len(ent["weight_assets"]),
        "trit_matmuls": diag.get("trit_matmul", 0),
    }
    return ent


def to_mermaid(ent):
    lines = ["erDiagram"]
    lines.append("    MODULE ||--o{ FUNCTION : contains")
    lines.append("    MODULE ||--o{ MEMORY_ARENA : declares")
    lines.append("    MODULE ||--o{ WEIGHT_ASSET : maps")
    lines.append("    FUNCTION ||--|{ IR_NODE : contains")
    lines.append("    FUNCTION }o--|| MEMORY_ARENA : allocates_in")
    lines.append("    IR_NODE }o--o| FUSED_KERNEL : grouped_into")
    lines.append("    MEMORY_ARENA ||--o{ MEMORY_SLICE : partitions")
    lines.append("    TENSOR_DESCRIPTOR ||--|| MEMORY_SLICE : offset_in")
    lines.append("    TENSOR_DESCRIPTOR }|--|| DATA_TYPE : typed_as")
    lines.append("    WEIGHT_ASSET ||--|{ PACKED_BLOCK : stored_as")
    lines.append("    DATA_TYPE ||--o| PACKING_RULE : governed_by")
    lines.append("")
    lines.append("    MODULE {")
    lines.append("        uuid module_id PK")
    lines.append("        string name")
    lines.append("        string target_arch")
    lines.append("        int cache_line_bytes")
    lines.append("    }")
    lines.append("    FUNCTION {")
    lines.append("        uuid function_id PK")
    lines.append("        uuid module_id FK")
    lines.append("        uuid arena_id FK")
    lines.append("        string identifier")
    lines.append("        bool is_pure")
    lines.append("    }")
    lines.append("    MEMORY_ARENA {")
    lines.append("        uuid arena_id PK")
    lines.append("        bigint total_capacity_bytes")
    lines.append("        bigint peak_used_bytes")
    lines.append("    }")
    lines.append("    MEMORY_SLICE {")
    lines.append("        uuid slice_id PK")
    lines.append("        bigint byte_offset")
    lines.append("        bigint byte_length")
    lines.append("        int birth_step")
    lines.append("        int death_step")
    lines.append("    }")
    lines.append("    IR_NODE {")
    lines.append("        uuid node_id PK")
    lines.append("        string op_code")
    lines.append("        int topological_order")
    lines.append("    }")
    lines.append("    FUSED_KERNEL {")
    lines.append("        uuid kernel_id PK")
    lines.append("        string symbol_name")
    lines.append("        string simd_strategy")
    lines.append("    }")
    lines.append("    DATA_TYPE {")
    lines.append("        uuid dtype_id PK")
    lines.append("        string name")
    lines.append("        float bit_width")
    lines.append("        bool requires_bit_packing")
    lines.append("    }")
    lines.append("    PACKING_RULE {")
    lines.append("        uuid rule_id PK")
    lines.append("        int elements_per_word")
    lines.append("        int container_bits")
    lines.append("    }")
    lines.append("    WEIGHT_ASSET {")
    lines.append("        uuid asset_id PK")
    lines.append("        string file_path")
    lines.append("        bool zero_copy_mmap")
    lines.append("    }")
    lines.append("    TENSOR_DESCRIPTOR {")
    lines.append("        uuid tensor_id PK")
    lines.append("        json shape")
    lines.append("        string dtype")
    lines.append("    }")
    return "\n".join(lines)
