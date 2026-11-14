"""Recursive-descent parser for Veloce."""

from . import ast
from .errors import ParseError


class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    # -- token helpers ----------------------------------------------------
    def peek(self, k=0):
        j = self.i + k
        if j >= len(self.toks):
            return self.toks[-1]
        return self.toks[j]

    def at(self, kind, value=None):
        t = self.peek()
        if t.kind != kind:
            return False
        if value is not None and t.value != value:
            return False
        return True

    def at_op(self, value):
        return self.at("OP", value)

    def at_kw(self, value):
        return self.at("KW", value)

    def advance(self):
        t = self.toks[self.i]
        if self.i < len(self.toks) - 1:
            self.i += 1
        return t

    def expect(self, kind, value=None):
        if not self.at(kind, value):
            t = self.peek()
            want = value if value is not None else kind
            raise ParseError(f"expected {want!r} but found {t.value!r}", t.line, t.col)
        return self.advance()

    def expect_op(self, value):
        return self.expect("OP", value)

    def skip_newlines(self):
        while self.at("NEWLINE") or self.at_op(";"):
            self.advance()

    # -- entry ------------------------------------------------------------
    def parse_module(self):
        self.skip_newlines()
        name = "main"
        if self.at_kw("module"):
            self.advance()
            name = self.expect("IDENT").value
        self.skip_newlines()
        dims = {}
        target_arch, cache_line = "native", 64
        while self.at("IDENT") and self.peek().value in ("dim", "target", "cache"):
            kw = self.peek().value
            if kw == "dim":
                self.advance()
                dname = self.expect("IDENT").value
                self.expect_op("=")
                dval = self.expect("INT").value
                dims[dname] = dval
            elif kw == "target":
                self.advance()
                target_arch = self.expect("STRING").value
            elif kw == "cache":
                self.advance()
                cache_line = self.expect("INT").value
            self.skip_newlines()
        funcs = []
        while not self.at("EOF"):
            funcs.append(self.parse_fn())
            self.skip_newlines()
        return ast.Module(name=name, functions=funcs, dims=dims,
                          target_arch=target_arch, cache_line_bytes=cache_line)

    # -- functions --------------------------------------------------------
    def parse_fn(self):
        line, col = self.peek().line, self.peek().col
        is_extern = False
        if self.at_kw("extern"):
            is_extern = True
            self.advance()
        self.expect("KW", "fn")
        name = self.expect("IDENT").value
        self.expect_op("(")
        self.skip_newlines()
        params = []
        if not self.at_op(")"):
            while True:
                self.skip_newlines()
                rw = False
                if self.at_kw("rw"):
                    self.advance()
                    rw = True
                pname = self.expect("IDENT").value
                self.expect_op(":")
                ptype = self.parse_type()
                params.append(ast.Param(name=pname, type=ptype, rw=rw))
                self.skip_newlines()
                if self.at_op(","):
                    self.advance()
                    self.skip_newlines()
                    continue
                break
        self.skip_newlines()
        self.expect_op(")")
        ret_types = []
        if self.at_op("->"):
            self.advance()
            ret_types.append(self.parse_type())
            while self.at_op(","):
                self.advance()
                ret_types.append(self.parse_type())
        fn = ast.FnDecl(line=line, col=col, name=name, params=params,
                        ret_types=ret_types, is_extern=is_extern)
        if is_extern:
            # body omitted; ends at newline
            return fn
        self.skip_newlines()
        fn.body = self.parse_block()
        return fn

    def parse_type(self):
        t = self.peek()
        if t.kind != "DTYPE":
            raise ParseError(f"expected a type, found {t.value!r}", t.line, t.col)
        self.advance()
        scalar = ast.ScalarType(t.value)
        if self.at_op("["):
            self.advance()
            dims = []
            while True:
                dt = self.peek()
                if dt.kind == "INT":
                    dims.append(ast.Dim(int(dt.value)))
                    self.advance()
                elif dt.kind == "IDENT":
                    dims.append(ast.Dim(dt.value))
                    self.advance()
                else:
                    raise ParseError(f"expected dimension, found {dt.value!r}", dt.line, dt.col)
                if self.at_op(","):
                    self.advance()
                    continue
                break
            self.expect_op("]")
            return ast.TensorType(dtype=scalar, dims=tuple(dims))
        return scalar

    # -- blocks & statements ---------------------------------------------
    def parse_block(self):
        self.expect_op("{")
        self.skip_newlines()
        stmts = []
        while not self.at_op("}"):
            if self.at("EOF"):
                raise ParseError("unexpected end of file, expected '}'",
                                 self.peek().line, self.peek().col)
            stmts.append(self.parse_stmt())
            self.skip_newlines()
        self.expect_op("}")
        return stmts

    def parse_stmt(self):
        t = self.peek()
        if self.at_kw("let") or self.at_kw("const"):
            return self.parse_let()
        if self.at_kw("return"):
            return self.parse_return()
        if self.at_kw("if"):
            return self.parse_if()
        if self.at_kw("while"):
            return self.parse_while()
        if self.at_kw("break"):
            self.advance()
            return ast.BreakStmt(line=t.line, col=t.col)
        if self.at_kw("continue"):
            self.advance()
            return ast.ContinueStmt(line=t.line, col=t.col)
        return self.parse_expr_or_assign()

    def parse_let(self):
        t = self.advance()  # let / const
        mut = t.value == "let"  # `const` is immutable, `let` is mutable
        names = [self.expect("IDENT").value]
        while self.at_op(","):
            self.advance()
            names.append(self.expect("IDENT").value)
        type_ann = None
        if self.at_op(":"):
            self.advance()
            type_ann = self.parse_type()
        self.expect_op("=")
        expr = self.parse_expr()
        return ast.LetStmt(line=t.line, col=t.col, names=names, mut=mut,
                           type_ann=type_ann, expr=expr)

    def parse_return(self):
        t = self.advance()
        exprs = []
        if not (self.at("NEWLINE") or self.at_op("}") or self.at("EOF")):
            exprs.append(self.parse_expr())
            while self.at_op(","):
                self.advance()
                exprs.append(self.parse_expr())
        return ast.ReturnStmt(line=t.line, col=t.col, exprs=exprs)

    def parse_if(self):
        t = self.advance()
        cond = self.parse_expr()
        self.skip_newlines()
        then_body = self.parse_block()
        else_body = []
        save = self.i
        self.skip_newlines()
        if self.at_kw("else"):
            self.advance()
            self.skip_newlines()
            if self.at_kw("if"):
                else_body = [self.parse_if()]
            else:
                else_body = self.parse_block()
        else:
            self.i = save
        return ast.IfStmt(line=t.line, col=t.col, cond=cond,
                          then_body=then_body, else_body=else_body)

    def parse_while(self):
        t = self.advance()
        cond = self.parse_expr()
        self.skip_newlines()
        body = self.parse_block()
        return ast.WhileStmt(line=t.line, col=t.col, cond=cond, body=body)

    def parse_expr_or_assign(self):
        t = self.peek()
        targets = [self.parse_expr()]
        while self.at_op(","):
            self.advance()
            targets.append(self.parse_expr())
        if self.at_op("=") or self.at_op("+=") or self.at_op("-=") or \
           self.at_op("*=") or self.at_op("/="):
            op = self.advance().value
            rhs = self.parse_expr()
            if op != "=":
                # desugar  a += b  ->  a = a + b
                lhs = targets[0]
                rhs = ast.BinOp(line=t.line, col=t.col, op=op[0], left=lhs, right=rhs)
            return ast.AssignStmt(line=t.line, col=t.col, targets=targets, expr=rhs)
        if len(targets) != 1:
            raise ParseError("comma-separated expressions are only valid in an assignment",
                             t.line, t.col)
        return ast.ExprStmt(line=t.line, col=t.col, expr=targets[0])

    # -- expressions ------------------------------------------------------
    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.at_op("||"):
            t = self.advance()
            right = self.parse_and()
            left = ast.BinOp(line=t.line, col=t.col, op="||", left=left, right=right)
        return left

    def parse_and(self):
        left = self.parse_equality()
        while self.at_op("&&"):
            t = self.advance()
            right = self.parse_equality()
            left = ast.BinOp(line=t.line, col=t.col, op="&&", left=left, right=right)
        return left

    def parse_equality(self):
        left = self.parse_comparison()
        while self.at_op("==") or self.at_op("!="):
            t = self.advance()
            right = self.parse_comparison()
            left = ast.BinOp(line=t.line, col=t.col, op=t.value, left=left, right=right)
        return left

    def parse_comparison(self):
        left = self.parse_add()
        while self.at_op("<") or self.at_op(">") or self.at_op("<=") or self.at_op(">="):
            t = self.advance()
            right = self.parse_add()
            left = ast.BinOp(line=t.line, col=t.col, op=t.value, left=left, right=right)
        return left

    def parse_add(self):
        left = self.parse_mul()
        while self.at_op("+") or self.at_op("-"):
            t = self.advance()
            right = self.parse_mul()
            left = ast.BinOp(line=t.line, col=t.col, op=t.value, left=left, right=right)
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.at_op("*") or self.at_op("/") or self.at_op("//") or \
                self.at_op("%") or self.at_op("@"):
            t = self.advance()
            right = self.parse_unary()
            left = ast.BinOp(line=t.line, col=t.col, op=t.value, left=left, right=right)
        return left

    def parse_unary(self):
        if self.at_op("-") or self.at_op("!") or self.at_op("+"):
            t = self.advance()
            operand = self.parse_unary()
            if t.value == "+":
                return operand
            return ast.UnaryOp(line=t.line, col=t.col, op=t.value, operand=operand)
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_primary()
        while self.at_op("["):
            t = self.advance()
            first = self.parse_expr()
            items = [first]
            while self.at_op(","):
                self.advance()
                items.append(self.parse_expr())
            self.expect_op("]")
            index = items[0] if len(items) == 1 else ast.TupleExpr(
                line=t.line, col=t.col, items=items)
            expr = ast.Index(line=t.line, col=t.col, base=expr, index=index)
        return expr

    def parse_primary(self):
        t = self.peek()
        if t.kind == "INT":
            self.advance()
            return ast.IntLit(line=t.line, col=t.col, value=t.value)
        if t.kind == "FLOAT":
            self.advance()
            return ast.FloatLit(line=t.line, col=t.col, value=t.value)
        if t.kind == "STRING":
            self.advance()
            return ast.StrLit(line=t.line, col=t.col, value=t.value)
        if t.kind == "DTYPE":
            # dtype used as a cast constructor:  f32(expr)
            self.advance()
            scalar = ast.ScalarType(t.value)
            if self.at_op("("):
                self.advance()
                inner = self.parse_expr()
                self.expect_op(")")
                return ast.Cast(line=t.line, col=t.col, expr=inner, dtype=scalar)
            raise ParseError(f"type {t.value!r} not valid in expression position",
                             t.line, t.col)
        if self.at_kw("true") or self.at_kw("false"):
            self.advance()
            return ast.BoolLit(line=t.line, col=t.col, value=(t.value == "true"))
        if self.at_kw("for"):
            raise ParseError("imperative `for` loops are not part of Veloce; "
                             "use declarative tensor operations", t.line, t.col)
        if t.kind == "IDENT":
            self.advance()
            name = t.value
            if self.at_op("("):
                self.advance()
                args = []
                kwargs = {}
                if not self.at_op(")"):
                    while True:
                        # keyword arg  name=expr
                        if self.peek().kind == "IDENT" and self.peek(1).kind == "OP" \
                                and self.peek(1).value == "=":
                            kname = self.advance().value
                            self.advance()  # '='
                            kwargs[kname] = self.parse_expr()
                        else:
                            args.append(self.parse_expr())
                        if self.at_op(","):
                            self.advance()
                            continue
                        break
                self.expect_op(")")
                return ast.Call(line=t.line, col=t.col, func=name, args=args, kwargs=kwargs)
            return ast.Var(line=t.line, col=t.col, name=name)
        if self.at_op("("):
            self.advance()
            inner = self.parse_expr()
            # tuple in parens: (a, b)
            if self.at_op(","):
                items = [inner]
                while self.at_op(","):
                    self.advance()
                    items.append(self.parse_expr())
                self.expect_op(")")
                return ast.TupleExpr(line=t.line, col=t.col, items=items)
            self.expect_op(")")
            return inner
        raise ParseError(f"unexpected token {t.value!r}", t.line, t.col)


def parse(tokens):
    return Parser(tokens).parse_module()
