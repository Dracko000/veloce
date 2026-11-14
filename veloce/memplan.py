"""Static memory arena planner.

Implements FR-3.1/3.2: compute exact tensor lifetimes across a function's
computation graph and assign non-overlapping tensor activations the *same*
physical byte offsets inside a pre-allocated static arena.

The planner walks a function body, assigning every *materialised* tensor (a
`let`-bound tensor value or a call result such as `matmul(...)`) a lifetime
``(birth_step, death_step)`` measured in statement order.  A greedy
first-fit linear scan then places each tensor into a free slot whose previous
occupant has already died, reusing buffers so that peak activation memory is
bounded by the largest simultaneously-live set rather than the sum of all
activations.

Offsets are stored on the AST nodes themselves (``node.arena_offset_words``,
``node.arena_size_words``) so code generation can emit them directly.
"""

from . import ast

ALIGN_WORDS = 16  # 64-byte alignment (16 x f32 words)


def _align(n, a=ALIGN_WORDS):
    return (n + a - 1) // a * a


class Allocation:
    __slots__ = ("node", "size_words", "birth", "death", "offset_words")

    def __init__(self, node, size_words, birth, death):
        self.node = node
        self.size_words = size_words
        self.birth = birth
        self.death = death
        self.offset_words = None


class ArenaPlan:
    def __init__(self):
        self.allocations = []
        self.peak_words = 0
        self.naive_words = 0  # sum of all activation sizes (no reuse)

    def report(self):
        saved = 1.0 - (self.peak_words / self.naive_words) if self.naive_words else 0.0
        return {
            "peak_words": self.peak_words,
            "peak_bytes": self.peak_words * 4,
            "naive_bytes": self.naive_words * 4,
            "reuse_savings_pct": round(saved * 100, 1),
        }


class Planner:
    def __init__(self):
        self.step = 0
        self._uses = {}  # var name -> last step used

    # ---- entry ----------------------------------------------------------
    def plan_function(self, fn):
        plan = ArenaPlan()
        self.step = 0
        self._uses = {}
        allocs = []

        # Flatten statement sequence to compute step numbers & last uses.
        self._collect(fn.body, allocs)

        if not allocs:
            fn.arena_words = 1
            fn.arena_plan = plan
            return plan

        # Resolve death steps: for a named tensor, last reference; for temps,
        # the step at which they were created (single-expression lifetime).
        for a in allocs:
            if isinstance(a.node, ast.LetStmt):
                # death = last step any of its names is referenced
                last = a.birth
                for nm in a.node.names:
                    last = max(last, self._uses.get(nm, a.birth))
                a.death = last
            else:
                a.death = a.birth  # temp lives only within its statement

        # Greedy first-fit reuse.
        self._place(allocs, plan)
        plan.allocations = allocs
        plan.naive_words = sum(a.size_words for a in allocs)
        fn.arena_words = max(1, plan.peak_words)
        fn.arena_plan = plan
        return plan

    # ---- traversal ------------------------------------------------------
    def _collect(self, stmts, allocs):
        for s in stmts:
            self.step += 1
            if isinstance(s, ast.LetStmt):
                self._mark_uses(s.expr)
                t = getattr(s.expr, "ctype", None)
                if isinstance(t, ast.TensorType):
                    words = self._words(t)
                    a = Allocation(s, words, self.step, self.step)
                    allocs.append(a)
                    s.arena_offset_words = None  # filled later
                # subexpression tensor temps (the root expr is owned by the
                # LetStmt slot itself, so skip allocating a duplicate for it)
                self._collect_temps(s.expr, allocs, root=isinstance(t, ast.TensorType))
                # binding names become "born" now; their uses tracked later
                for nm in s.names:
                    self._uses.setdefault(nm, self.step)
                    self._uses[nm] = max(self._uses[nm], self.step)
            elif isinstance(s, ast.AssignStmt):
                self._mark_uses(s.expr)
                self._collect_temps(s.expr, allocs)
                for tgt in s.targets:
                    if isinstance(tgt, ast.Var):
                        self._uses[tgt.name] = max(self._uses.get(tgt.name, self.step), self.step)
            elif isinstance(s, ast.ReturnStmt):
                for e in s.exprs:
                    self._mark_uses(e)
                    self._collect_temps(e, allocs)
            elif isinstance(s, ast.IfStmt):
                self._mark_uses(s.cond)
                self._collect(s.then_body, allocs)
                self._collect(s.else_body, allocs)
            elif isinstance(s, ast.WhileStmt):
                self._mark_uses(s.cond)
                self._collect(s.body, allocs)
            elif isinstance(s, ast.ExprStmt):
                self._mark_uses(s.expr)
                self._collect_temps(s.expr, allocs)

    def _mark_uses(self, e):
        if e is None:
            return
        if isinstance(e, ast.Var):
            self._uses[e.name] = max(self._uses.get(e.name, 0), self.step)
        elif isinstance(e, ast.Index):
            self._mark_uses(e.base)
            self._mark_uses(e.index)
        elif isinstance(e, ast.UnaryOp):
            self._mark_uses(e.operand)
        elif isinstance(e, ast.BinOp):
            self._mark_uses(e.left)
            self._mark_uses(e.right)
        elif isinstance(e, ast.Call):
            for a in e.args:
                self._mark_uses(a)
            for v in e.kwargs.values():
                self._mark_uses(v)
        elif isinstance(e, ast.Cast):
            self._mark_uses(e.expr)
        elif isinstance(e, ast.TupleExpr):
            for i in e.items:
                self._mark_uses(i)

    def _collect_temps(self, e, allocs, root=False):
        """Register materialised call-result temporaries (e.g. matmul output).

        ``root=True`` means the caller (a LetStmt) already owns a slot for this
        exact expression, so we do not allocate a second one for it -- but its
        children may still need temporaries.
        """
        if e is None:
            return
        if isinstance(e, ast.Call):
            for a in e.args:
                self._collect_temps(a, allocs)
            t = getattr(e, "ctype", None)
            if not root and isinstance(t, ast.TensorType) and getattr(e, "lower", None) in (
                    "matmul_f32", "matmul_trit", "matmul_trit_s", "matmul_i4",
                    "matmul_i4g", "matmul_f16wt", "rmsnorm", "silu", "relu",
                    "softmax", "rope", "rope_heads", "embedding", "transpose",
                    "rowmax", "embedding_seq", "meanpool", "zeros"):
                words = self._words(t)
                a = Allocation(e, words, self.step, self.step)
                allocs.append(a)
        elif isinstance(e, ast.BinOp):
            self._collect_temps(e.left, allocs)
            self._collect_temps(e.right, allocs)
            t = getattr(e, "ctype", None)
            if not root and isinstance(t, ast.TensorType):
                words = self._words(t)
                a = Allocation(e, words, self.step, self.step)
                allocs.append(a)
        elif isinstance(e, ast.UnaryOp):
            self._collect_temps(e.operand, allocs)
        elif isinstance(e, ast.Cast):
            self._collect_temps(e.expr, allocs)
        elif isinstance(e, ast.Index):
            self._collect_temps(e.base, allocs)
        elif isinstance(e, ast.TupleExpr):
            for i in e.items:
                self._collect_temps(i, allocs)

    def _words(self, tensor_type):
        n = tensor_type.numel()
        if n is None:
            raise ValueError("cannot plan arena for a symbolic-shaped tensor")
        return _align(n)

    # ---- allocator ------------------------------------------------------
    def _place(self, allocs, plan):
        free = []  # list of (offset, size)
        top = 0
        order = sorted(range(len(allocs)), key=lambda i: (allocs[i].birth, -allocs[i].size_words))
        live = []  # indices currently placed
        for idx in order:
            a = allocs[idx]
            # free any slots whose occupant died before this birth
            still_live = []
            for j in live:
                if allocs[j].death < a.birth:
                    free.append((allocs[j].offset_words, allocs[j].size_words))
                else:
                    still_live.append(j)
            live = still_live
            # first-fit
            free.sort()
            spot = None
            for k, (off, sz) in enumerate(free):
                if sz >= a.size_words:
                    spot = (off, sz)
                    del free[k]
                    break
            if spot is None:
                a.offset_words = top
                top += _align(a.size_words)
            else:
                a.offset_words = spot[0]
                # return the tail if larger
                if spot[1] > a.size_words:
                    free.append((spot[0] + _align(a.size_words), spot[1] - a.size_words))
            live.append(idx)
            # annotate node
            try:
                a.node.arena_offset_words = a.offset_words
                a.node.arena_size_words = a.size_words
            except Exception:
                pass
        plan.peak_words = top


def plan_module(module):
    planner = Planner()
    plans = {}
    for fn in module.functions:
        if fn.is_extern:
            continue
        plans[fn.name] = planner.plan_function(fn)
    return plans
