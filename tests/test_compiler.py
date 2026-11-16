#!/usr/bin/env python3
"""Veloce compiler test suite (stdlib unittest).

Run:  python3 tests/test_compiler.py -v
"""
import os
import struct
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import veloce  # noqa: E402
from veloce import weights  # noqa: E402
from veloce.errors import (  # noqa: E402
    LexError, ParseError, TypeError_, SemanticError,
)

RUNTIME = os.path.join(ROOT, "runtime")
CC = os.environ.get("CC", "cc")


def build_and_run(vel_src, driver_src, extra_arrays="", name="t"):
    """Compile a Veloce source + a C driver, link, run, return stdout."""
    res = veloce.compile_source(vel_src, module_name=name)
    cpath = f"/tmp/{name}.gen.c"
    with open(cpath, "w") as f:
        f.write(res.c_source)
    drv = f"/tmp/{name}.drv.c"
    with open(drv, "w") as f:
        f.write('#include "veloce_runtime.h"\n#include <stdio.h>\n')
        f.write(extra_arrays)
        f.write("\n")
        f.write(driver_src)
    exe = f"/tmp/{name}.bin"
    cmd = [CC, "-std=c11", "-O3", "-march=native", "-ffast-math",
           "-I", RUNTIME, drv, cpath, "-lm", "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("compile failed:\n" + r.stderr + "\n--- C ---\n" + res.c_source)
    r = subprocess.run([exe], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("run failed: " + r.stderr)
    return r.stdout, res


class TestFrontEnd(unittest.TestCase):
    def test_lex(self):
        toks = veloce.tokenize("fn f(a: i32) -> i32 { return a }\n")
        kinds = [t.kind for t in toks]
        self.assertIn("KW", kinds)
        self.assertEqual(toks[-1].kind, "EOF")

    def test_lex_error(self):
        with self.assertRaises(LexError):
            veloce.tokenize("let x = $")

    def test_parse_fn(self):
        mod = veloce.parse(veloce.tokenize(
            "fn add(a: i32, b: i32) -> i32 { return a + b }\n"))
        self.assertEqual(len(mod.functions), 1)
        self.assertEqual(mod.functions[0].name, "add")

    def test_parse_error(self):
        with self.assertRaises(ParseError):
            veloce.parse(veloce.tokenize("fn f( { }"))

    def test_no_for_loops(self):
        with self.assertRaises(ParseError):
            veloce.compile_source("fn f() -> i32 { for x in y { } return 0 }")


class TestSemantics(unittest.TestCase):
    def test_undefined_var(self):
        with self.assertRaises(SemanticError):
            veloce.compile_source("fn f() -> i32 { return zzz }")

    def test_shape_mismatch(self):
        src = ("dim N = 4\n"
               "fn f(a: f32[3,4], b: f32[2,4]) -> f32[3,4] { return a + b }")
        with self.assertRaises(TypeError_):
            veloce.compile_source(src)

    def test_matmul_dim_mismatch(self):
        src = ("dim A = 3\ndim B = 4\ndim C = 5\n"
               "fn f(x: f32[A,B], w: f32[C,C]) -> f32[A,C] { return matmul(x, w) }")
        with self.assertRaises(TypeError_):
            veloce.compile_source(src)

    def test_immutable_assign(self):
        src = "fn f() -> i32 { const a = 1\n a = 2\n return a }"
        with self.assertRaises(SemanticError):
            veloce.compile_source(src)

    def test_trit_matmul_requires_trit(self):
        src = ("dim A = 2\ndim B = 3\n"
               "fn f(x: f32[A,B], w: f32[B,B]) -> f32[A,B] { return matmul_trit(x, w) }")
        with self.assertRaises(TypeError_):
            veloce.compile_source(src)

    def test_undeclared_dim(self):
        with self.assertRaises(veloce.VeloceError):
            veloce.compile_source("fn f(x: f32[Q, 4]) -> f32[Q,4] { return x }")


class TestCodegen(unittest.TestCase):
    def test_hello_runs(self):
        src = ("fn add_scale(a: i32, b: i32, s: f32) -> f32 {\n"
               "  let sum = a + b\n  return f32(sum) * s\n}\n")
        out, _ = build_and_run(
            src,
            'extern float veloce_add_scale(int,int,float);\n'
            'int main(void){ printf("%.3f\\n", veloce_add_scale(3,4,2.5)); return 0; }',
            name="hello")
        self.assertAlmostEqual(float(out.strip()), 17.5, places=4)

    def test_if_while_multi_return(self):
        src = ("fn count(n: i32) -> i32, i32 {\n"
               "  let i = 0\n  let s = 0\n"
               "  while i < n {\n    if i % 2 == 0 { s = s + i }\n    i = i + 1\n  }\n"
               "  return s, i\n}\n")
        out, _ = build_and_run(
            src,
            'extern void veloce_count(int,int*,int*);\n'
            'int main(void){ int s,i; veloce_count(10,&s,&i);'
            ' printf("%d %d\\n", s, i); return 0; }',
            name="loop")
        s, i = [int(v) for v in out.split()]
        self.assertEqual((s, i), (0 + 2 + 4 + 6 + 8, 10))

    def test_argmax(self):
        src = ("dim N = 5\n"
               "fn pick(x: f32[N]) -> i32 {\n  return argmax(x)\n}\n")
        arr = "static float XV[5] = {0.1f, 3.2f, -1.0f, 7.5f, 2.0f};\n"
        out, _ = build_and_run(
            src,
            'extern int veloce_pick(float*);\n'
            'int main(void){ printf("%d\\n", veloce_pick(XV)); return 0; }',
            extra_arrays=arr, name="argmax")
        self.assertEqual(int(out.strip()), 3)

    def test_arena_reuse_reduces_peak(self):
        # A chain of same-shaped tensors should reuse one buffer.
        src = ("dim N = 64\n"
               "fn chain(a: f32[N], b: f32[N]) -> f32[N] {\n"
               "  let x0 = a + b\n  let x1 = x0 + b\n  let x2 = x1 + b\n"
               "  let x3 = x2 + b\n  let x4 = x3 + b\n  return x4\n}\n")
        res = veloce.compile_source(src)
        plan = res.memory["chain"]
        rpt = plan.report()
        self.assertGreater(rpt["reuse_savings_pct"], 40.0,
                           f"expected strong reuse, got {rpt}")


class TestPacking(unittest.TestCase):
    def test_trit_roundtrip(self):
        import random
        random.seed(7)
        vals = [random.choice([-1, 0, 1]) for _ in range(1000)]
        packed = weights.pack_trits(vals)
        back = weights.unpack_trits(packed, len(vals))
        self.assertEqual(back, vals)

    def test_i4_roundtrip(self):
        import random
        random.seed(11)
        vals = [random.randint(-8, 7) for _ in range(1001)]
        packed = weights.pack_i4(vals)
        self.assertEqual(len(packed), (len(vals) + 1) // 2)
        back = weights.unpack_i4(packed, len(vals))
        self.assertEqual(back, vals)

    def test_i4_pack_layout(self):
        # low nibble first, sign-extended
        self.assertEqual(list(weights.pack_i4([1, 2])), [0x21])
        # low nibble first: -1 -> 0xF in low, -8 -> 0x8 in high => 0x8F
        self.assertEqual(list(weights.pack_i4([-1, -8])), [0x8F])
        self.assertEqual(list(weights.unpack_i4(bytes([0x21]), 2)), [1, 2])
        self.assertEqual(list(weights.unpack_i4(bytes([0x8F]), 2)), [-1, -8])

    def test_i4_matmul_codegen(self):
        src = """module t
dim M = 2
dim K = 64
dim N = 8
fn f(x: f32[M, K], w: i4[N, K]) -> f32[M, N] {
    let y = matmul_i4(x, w)
    return relu(y)
}
"""
        c = veloce.compile_source(src, module_name="t").c_source
        self.assertIn("veloce_matmul_i4", c)

    def test_i4g_group_size_matches_scales(self):
        """matmul_i4g must hand the runtime the group size implied by the
        declared scale shape, not a floor division that disagrees with it."""
        src = """module t
dim M = 2
dim K = 64
dim N = 8
dim NG = 16
fn f(x: f32[M, K], w: i4[N, K], s: f32[N, NG]) -> f32[M, N] {
    return matmul_i4g(x, w, s)
}
"""
        c = veloce.compile_source(src, module_name="t").c_source
        self.assertIn("veloce_matmul_i4g(", c)
        self.assertIn(", 64, 8, 4);", c)   # K=64, N=8, group size 64/16=4

    def test_i4g_rejects_ragged_group_count(self):
        """A group count that does not divide K has no single group size, so it
        must be rejected instead of silently scaling the wrong slices."""
        src = """module t
dim M = 2
dim K = 100
dim N = 5
dim NG = 7
fn f(x: f32[M, K], w: i4[N, K], s: f32[N, NG]) -> f32[M, N] {
    return matmul_i4g(x, w, s)
}
"""
        with self.assertRaises(TypeError_):
            veloce.compile_source(src, module_name="t")

    def test_trit_s_codegen(self):
        src = """module t
dim M = 2
dim K = 64
dim N = 8
fn f(x: f32[M, K], w: trit[N, K], s: f32[N]) -> f32[M, N] {
    return matmul_trit_s(x, w, s)
}
"""
        c = veloce.compile_source(src, module_name="t").c_source
        self.assertIn("veloce_matmul_trit_s(", c)

    def test_i2_and_fp8_roundtrip(self):
        """The .veloce container carries the new dtypes zero-copy."""
        import numpy as np
        w2 = np.array([[-2, 1, 0, -1], [1, 1, -2, 0]], dtype=np.int8)
        wf = np.array([[1.0, -1.5, 0.25, 448.0]], dtype=np.float32)
        man = weights.write_veloce_model(
            "/tmp/t_i2fp8.veloce",
            [{"name": "a", "dtype": "i2", "data": w2},
             {"name": "b", "dtype": "fp8", "data": wf}])
        # 4 i2 values per byte, 1 byte per fp8 value
        self.assertEqual(man["tensors"][0]["nbytes"], 2)
        self.assertEqual(man["tensors"][1]["nbytes"], 4)
        f = weights.VeloceFile("/tmp/t_i2fp8.veloce")
        try:
            self.assertEqual(f.numpy("a").tolist(), w2.tolist())
            self.assertEqual(f.numpy("b").tolist(), wf.tolist())
            self.assertEqual(len(f.raw("a")), 2)   # zero-copy view
        finally:
            f.close()

    def test_i2_codegen(self):
        src = """module t
dim M = 2
dim K = 64
dim N = 8
fn f(x: f32[M, K], w: i2[N, K]) -> f32[M, N] {
    return matmul_i2(x, w)
}
fn g(x: f32[M, K], w: i2[N, K], s: f32[N]) -> f32[M, N] {
    return matmul_i2_s(x, w, s)
}
"""
        c = veloce.compile_source(src, module_name="t").c_source
        self.assertIn("veloce_matmul_i2(", c)
        self.assertIn("veloce_matmul_i2_s(", c)

    def test_fp8_codegen(self):
        src = """module t
dim M = 2
dim K = 64
dim N = 8
fn f(x: f32[M, K], w: fp8[N, K]) -> f32[M, N] {
    return matmul_fp8(x, w)
}
fn g(x: f32[M, K], w: fp8[N, K], s: f32[N]) -> f32[M, N] {
    return matmul_fp8_s(x, w, s)
}
"""
        c = veloce.compile_source(src, module_name="t").c_source
        self.assertIn("veloce_matmul_fp8(", c)
        self.assertIn("veloce_matmul_fp8_s(", c)

    def test_group_scaled_codegen(self):
        """`matmul_i2g`/`matmul_fp8g` lower to their own runtime symbols.

        The group SIZE is derived from the scale tensor's shape exactly like
        `matmul_i4g` (K // n_groups, never a floor that disagrees with the
        runtime), and all three share one lowering path.
        """
        src = """module t
dim M = 2
dim K = 64
dim N = 8
dim NG = 16
fn a(x: f32[M, K], w: i2[N, K], s: f32[N, NG]) -> f32[M, N] {
    return matmul_i2g(x, w, s)
}
fn b(x: f32[M, K], w: fp8[N, K], s: f32[N, NG]) -> f32[M, N] {
    return matmul_fp8g(x, w, s)
}
fn c(x: f32[M, K], w: i4[N, K], s: f32[N, NG]) -> f32[M, N] {
    return matmul_i4g(x, w, s)
}
"""
        c = veloce.compile_source(src, module_name="t").c_source
        self.assertIn("veloce_matmul_i2g(", c)
        self.assertIn("veloce_matmul_fp8g(", c)
        self.assertIn("veloce_matmul_i4g(", c)
        # K=64 over NG=16 groups -> group size 4, passed as the last argument.
        for sym in ("veloce_matmul_i2g(", "veloce_matmul_fp8g(",
                    "veloce_matmul_i4g("):
            line = next(ln for ln in c.splitlines() if sym in ln)
            self.assertTrue(line.rstrip().endswith("4);"), line)

    def test_group_scaled_rejects_wrong_dtype(self):
        """A group-scaled kernel demands its own weight dtype."""
        for fn, dt, wrong in (("matmul_i2g", "i2", "i4"),
                              ("matmul_fp8g", "fp8", "i2")):
            src = f"""module t
dim M = 2
dim K = 64
dim N = 8
dim NG = 16
fn f(x: f32[M, K], w: {wrong}[N, K], s: f32[N, NG]) -> f32[M, N] {{
    return {fn}(x, w, s)
}}
"""
            with self.assertRaises(TypeError_):
                veloce.compile_source(src, module_name="t")

    def test_group_scaled_rejects_ragged_groups(self):
        """The group count must divide K, for every group-scaled kernel."""
        for fn, dt in (("matmul_i4g", "i4"), ("matmul_i2g", "i2"),
                       ("matmul_fp8g", "fp8")):
            src = f"""module t
dim M = 2
dim K = 100
dim N = 4
dim NG = 7
fn f(x: f32[M, K], w: {dt}[N, K], s: f32[N, NG]) -> f32[M, N] {{
    return {fn}(x, w, s)
}}
"""
            with self.assertRaises(TypeError_) as cm:
                veloce.compile_source(src, module_name="t")
            self.assertIn("divide K exactly", str(cm.exception))

    def test_lowbit_matmul_rejects_wrong_dtype(self):
        """Each low-bit matmul demands its own storage dtype."""
        for fn, dt, wrong in (("matmul_i2", "i2", "i4"), ("matmul_fp8", "fp8", "i2")):
            src = f"""module t
dim M = 2
dim K = 64
dim N = 8
fn f(x: f32[M, K], w: {wrong}[N, K]) -> f32[M, N] {{
    return {fn}(x, w)
}}
"""
            with self.assertRaises(TypeError_):
                veloce.compile_source(src, module_name="t")

    def test_trit_size(self):
        vals = [1] * 1000
        packed = weights.pack_trits(vals)
        # 5 trits per byte -> 200 bytes (vs 1000 for int8, 2000 for fp16, 4000 f32)
        self.assertEqual(len(packed), 200)

    def test_matches_runtime_header(self):
        # The C encoder's format must agree with Python's for a random vector.
        import random
        random.seed(3)
        vals = [random.choice([-1, 0, 1]) for _ in range(50)]
        packed = weights.pack_trits(vals)
        cdecl = "static uint8_t P[%d] = {%s};\n" % (
            len(packed), ",".join(str(b) for b in packed))
        drv = ('extern int veloce_check(uint8_t*);\n'
               'int main(void){ printf("%d\\n", veloce_check(P)); return 0; }\n')
        # use a tiny veloce fn that unpacks via runtime and returns a checksum
        src = ("dim N = 50\n"
               "fn check(x: trit[N]) -> i32 {\n  return 0\n}\n")
        # Instead of relying on a veloce fn we compile a C helper via the runtime
        helper = ('#include "veloce_runtime.h"\n'
                  'int veloce_check(uint8_t*p){ int s=0; int8_t tmp[50];'
                  ' veloce_expand_trit(tmp,p,50); for(int i=0;i<50;i++) s+=tmp[i]*(i+1);'
                  ' return s; }')
        cpath = "/tmp/pk.helper.c"
        with open(cpath, "w") as f:
            f.write(helper)
        drvpath = "/tmp/pk.drv.c"
        with open(drvpath, "w") as f:
            f.write('#include "veloce_runtime.h"\n#include <stdio.h>\n')
            f.write(cdecl)
            f.write(drv)
        exe = "/tmp/pk.bin"
        r = subprocess.run([CC, "-std=c11", "-O2", "-I", RUNTIME, cpath, drvpath,
                            "-lm", "-o", exe], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = subprocess.run([exe], capture_output=True, text=True).stdout
        expect = sum(v * (i + 1) for i, v in enumerate(vals))
        self.assertEqual(int(out.strip()), expect)


class TestBenchmarkMetrics(unittest.TestCase):
    """Guard the throughput benchmark against the bugs that inflated its claims.

    The published "99% of streaming peak" came from two defects: a scalar
    1-thread probe compared against a 4-thread kernel, and dividing by T4_MS
    regardless of which run was fastest.  These tests pin the fixes.
    """

    def setUp(self):
        import importlib.util
        path = os.path.join(ROOT, "benchmarks", "bench_throughput.py")
        spec = importlib.util.spec_from_file_location("bench_throughput", path)
        self.bt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bt)

    def test_bandwidth_ratio_uses_matching_thread_count(self):
        """The probe and the kernel must be measured at the same thread count.

        Equality here is the property that was violated: a 1-thread probe vs a
        4-thread kernel can report a ratio above 100% of "peak".
        """
        import inspect
        src = inspect.getsource(self.bt.bench_membw)
        self.assertIn("threads", inspect.signature(self.bt.bench_membw).parameters)
        self.assertIn("num_threads({threads})", src)
        self.assertIn("schedule(static)", src)
        # and the caller must feed it the GEMV's own thread count
        main_src = inspect.getsource(self.bt.main)
        self.assertIn("bench_membw(args.quick, threads=gemv_threads)", main_src)
        self.assertIn('gemv_ms = gemv[f"T{gemv_threads}_MS"]', main_src)
        self.assertIn('gemv[f"T{gemv_threads}_MS"], attn', main_src)

    def test_probe_is_not_a_scalar_byte_sum(self):
        """The old probe was `for i: acc += a[i]`, which measured a scalar loop.

        A real ceiling has to be at least parallel; assert the probe body reads
        several bytes per iteration rather than one.
        """
        import inspect
        src = inspect.getsource(self.bt.bench_membw)
        self.assertIn("a[i+15]", src, "probe should read 16 bytes per step")
        self.assertNotIn("for (size_t i = 0; i < n; i++) acc += a[i];", src)


class TestVeloceFormat(unittest.TestCase):
    def test_roundtrip(self):
        import numpy as np
        w = np.array([[-1, 0, 1], [1, 1, -1]], dtype=np.int8)
        f = np.arange(6, dtype=np.float32).reshape(2, 3)
        h = np.array([1.5, -2.25, 3.0], dtype=np.float16)
        path = "/tmp/m.veloce"
        weights.save_veloce(path, [("w", w, "trit"), ("f", f, "f32"), ("h", h, "fp16")])
        vf = weights.VeloceFile(path)
        try:
            np.testing.assert_array_equal(vf.numpy("w"), w)
            np.testing.assert_array_equal(vf.numpy("f"), f)
            np.testing.assert_allclose(vf.numpy("h").astype(np.float32), h.astype(np.float32))
            # zero-copy: raw returns a memoryview
            self.assertIsInstance(vf.raw("w"), memoryview)
        finally:
            vf.close()

    def test_alignment(self):
        import numpy as np
        path = "/tmp/al.veloce"
        weights.save_veloce(path, [("a", np.zeros(10, dtype=np.float32), "f32"),
                                   ("b", np.zeros(33, dtype=np.int8), "trit")])
        vf = weights.VeloceFile(path)
        try:
            for e in vf.entries.values():
                self.assertEqual(e["offset"] % 64, 0)
        finally:
            vf.close()

    def test_model_card_roundtrip(self):
        import numpy as np
        from veloce import capabilities
        path = "/tmp/mc.veloce"
        card = capabilities.make_model_card(
            "unit-test", params={"ternary": 64},
            capabilities={"coding_agentic": {
                "status": "trained",
                "evidence": ["unit-test fixture"],
            }})
        weights.save_veloce(path, [("w", np.array([1, 0, -1], dtype=np.int8), "trit")],
                            model_card=card)
        vf = weights.VeloceFile(path)
        try:
            self.assertIsNotNone(vf.model_card)
            self.assertEqual(vf.model_card["name"], "unit-test")
            self.assertEqual(
                vf.model_card["capabilities"]["coding_agentic"]["status"],
                "trained")
            # Untouched pillars stay conservatively declared.
            self.assertEqual(
                vf.model_card["capabilities"]["world_understanding"]["status"],
                "declared")
            capabilities.validate_model_card(vf.model_card)
        finally:
            vf.close()

    def test_model_card_rejects_overclaim(self):
        from veloce import capabilities
        card = capabilities.make_model_card("bad")
        card["capabilities"]["specialist_knowledge"] = {
            "status": "verified", "evidence": []}
        with self.assertRaises(ValueError):
            capabilities.validate_model_card(card)
        with self.assertRaises(ValueError):
            capabilities.make_model_card(
                "bad", capabilities={"not_a_pillar": {"status": "declared"}})
        with self.assertRaises(ValueError):
            capabilities.make_model_card(
                "bad", capabilities={"coding_agentic": {"status": "maybe"}})


class TestDecoderModule(unittest.TestCase):
    """veloce.lmmodule renders the benchmark module; it must stay in sync."""

    def test_default_render_matches_example(self):
        """examples/midi_lm.vel is exactly the default config's rendering."""
        from veloce.lmmodule import decoder_module
        path = os.path.join(ROOT, "examples", "midi_lm.vel")
        with open(path) as f:
            on_disk = f.read()
        self.assertEqual(on_disk, decoder_module({}),
                         "examples/midi_lm.vel drifted from veloce/lmmodule.py; "
                         "run `make model-module`")

    def test_forward_params_follow_config(self):
        """Layer count, d_model and ffn must reach the rendered signature."""
        from veloce.lmmodule import decoder_module, param_names
        cfg = dict(name="t05b", V=256, D=2048, F=5632, layers=7)
        src = decoder_module(cfg)
        self.assertIn("module t05b", src)
        self.assertIn("dim D = 2048", src)
        self.assertIn("dim F = 5632", src)
        names = param_names(cfg)
        self.assertEqual(len(names), 1 + 7 * 8 + 2)
        self.assertEqual(names[0], "emb")
        self.assertEqual(names[-2:], ["nwf", "head"])
        self.assertIn("nw7", names)
        self.assertIn("w27", names)
        self.assertNotIn("nw8", names)
        fn = next(f for f in veloce.parse(veloce.tokenize(src)).functions
                  if f.name == "forward")
        # param_names() lists the *weight* tensors; `tok` is a scalar.
        tensors = [p.name for p in fn.params
                   if isinstance(p.type, veloce.ast_mod.TensorType)]
        self.assertEqual(tensors, names)

    def test_scale_defaults_and_override(self):
        from veloce.lmmodule import attn_scale
        self.assertAlmostEqual(attn_scale(dict(D=1024)), 1.0 / 32.0)
        self.assertAlmostEqual(attn_scale(dict(D=1024, scale=0.5)), 0.5)

    def test_renders_compile_to_c(self):
        """A non-default shape must actually compile, not just render."""
        from veloce.lmmodule import decoder_module
        cfg = dict(name="tiny_dm", V=64, D=32, F=64, layers=2)
        res = veloce.compile_source(decoder_module(cfg), module_name="tiny_dm")
        self.assertEqual(res.diagnostics["trit_matmul"], 8)
        self.assertIn("veloce_matmul_trit", res.c_source)

    def test_rejects_bad_config(self):
        from veloce.lmmodule import decoder_module
        with self.assertRaises(ValueError):
            decoder_module(dict(D=0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
