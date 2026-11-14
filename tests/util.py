"""Shared helpers for the Veloce test-suite.

Compiles a .vel file with the Veloce compiler, links the generated C against
a small C driver, runs it, and returns stdout.  Keeps everything stdlib-only.
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(ROOT, "runtime")
sys.path.insert(0, ROOT)

import veloce  # noqa: E402

CC = os.environ.get("CC", "cc")


def emit(vel_path):
    """Compile a .vel file to C source (string)."""
    return veloce.compile_file(vel_path)


def build_and_run(vel_source, driver_c, extra_flags=None, name="prog",
                  link_flags=None):
    """Write vel_source and driver_c to a temp dir, compile, link, run.

    `extra_flags` apply to both compile and link steps (e.g. -fopenmp).
    Returns (returncode, stdout, stderr, c_source).
    """
    work = tempfile.mkdtemp(prefix="veloce_test_")
    vel_path = os.path.join(work, name + ".vel")
    with open(vel_path, "w") as f:
        f.write(vel_source)
    res = veloce.compile_file(vel_path)
    gen_c = os.path.join(work, name + "_gen.c")
    with open(gen_c, "w") as f:
        f.write(res.c_source)
    drv_c = os.path.join(work, "driver.c")
    with open(drv_c, "w") as f:
        f.write(driver_c)

    flags = ["-std=c11", "-O2", "-march=native", "-pthread", "-I", RUNTIME,
             "-Wno-unused-parameter", "-Wno-unused-function"]
    if extra_flags:
        flags += extra_flags
    gen_o = os.path.join(work, "gen.o")
    drv_o = os.path.join(work, "driver.o")
    prog = os.path.join(work, "prog")

    for src, obj in ((gen_c, gen_o), (drv_c, drv_o)):
        r = subprocess.run([CC, *flags, "-c", src, "-o", obj],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"C compile failed for {src}:\n{r.stderr}")
    r = subprocess.run([CC, gen_o, drv_o, *flags, "-lm", "-o", prog],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"link failed:\n{r.stderr}")
    r = subprocess.run([prog], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr, res.c_source


def run_main(vel_source, main_body, includes="", decls="", extra_flags=None):
    """Convenience: build a driver whose main() is `main_body`.

    `decls` are extra top-level C declarations (e.g. extern prototypes);
    `includes` extra #include lines.  veloce_runtime.h is always available.
    """
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
{includes}
{decls}
int main(void) {{
{main_body}
    return 0;
}}
"""
    return build_and_run(vel_source, driver, extra_flags=extra_flags)
