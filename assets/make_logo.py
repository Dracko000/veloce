#!/usr/bin/env python3
"""Generate the Veloce mark.

The mark is not decoration: five bars = five trits = one packed byte, the exact
unit Veloce's trit packing uses (``5 trits -> 1 byte``).  Their heights form a
V, and each bar's *height* encodes its trit value: +1 tallest, 0 middle,
-1 shortest.  Read left to right the heights are (+, 0, -, 0, +), i.e. the
balanced ternary digit string "10-01".

Outputs (in assets/):
    logo.svg              primary lockup, dark background
    logo-light.svg        same, for light backgrounds
    logo-mark.svg         the mark alone (square, for avatars/favicons)
    logo-mark-light.svg   the mark alone, dark ink
    logo-wordmark.svg     the word only
    logo.png              1024x1024 social/marketing raster
    logo-256.png          small raster
    favicon.ico           multi-size icon for the docs site

Run:  python3 assets/make_logo.py
"""
import os
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))

# Palette: near-black slate + the amber of a 1.58-bit "velocity" accent,
# with a cool violet reserved for the negative trit so the ternary reads
# instantly (+=amber, 0=slate, -=violet).
BG_DARK = "#0d1117"
INK_ON_DARK = "#e6edf3"
INK_ON_LIGHT = "#0d1117"
AMBER = "#ffb020"
VIOLET = "#7c8cff"
SLATE = "#3d4757"          # tick/detail ink
MUTED = "#8b949e"          # tagline on dark: ~7:1 contrast on BG_DARK
MUTED_LIGHT = "#57606a"    # tagline on white
GRID = "#1c2333"

# The mark is three nested chevrons: a "V" for Veloce, repeated and offset the
# way motion streaks repeat, which is what "veloce" means.  Under it sit five
# ticks -- five trits, one packed byte -- the unit the whole language is built
# around.  Both ideas are load-bearing: neither is decoration.
# Read left to right: -1, 0, +1, 0, -1.  The tallest tick sits in the middle so
# the group carries even optical weight under the V (a tall tick at one end
# dragged the eye off-centre even though the group was geometrically centred).
# The sequence is not a palindrome, so it reads as data rather than a spinner.
TRITS = (-1, 0, +1, 0, -1)


# Vertical fraction of the box the V occupies, and how far the vertex drops.
# A shallower (wider-angle) V leaves a counter big enough to hold the trit
# ticks inside the mark, which is what makes the two ideas one object.
V_TOP = 0.115
V_BOTTOM = 0.585
V_DEPTH = 0.88


def chevron_path(size, inset=None, arm_depth=None):
    """The V stroke: from the top corners down to the vertex and back up."""
    s = size
    pad = s * 0.075
    top = s * V_TOP
    bottom = s * V_BOTTOM
    cx = s / 2
    depth = (bottom - top) * V_DEPTH
    return f"M {pad:.2f} {top:.2f} L {cx:.2f} {top + depth:.2f} L {s - pad:.2f} {top:.2f}"


def mark_ink_box(size):
    """Vertical extent of the mark's ink inside a `size` box, in box units."""
    s = size
    top = s * V_TOP
    vy = top + (s * V_BOTTOM - top) * V_DEPTH
    half = s * 0.125 / 2
    return top - half, vy + half


def mark_svg(size=100, ink_bg=None, dark=True):
    """A bold V, ink-centred in `size`.

    One object, one reading: the name.  A five-tick "one packed byte" motif was
    tried inside the counter and below the vertex; at every size that read as a
    separate UI progress indicator (or collided with the stroke at the vertex),
    which weakens the mark instead of encoding anything.  The trit idea lives in
    the code, not in the logo.
    """
    s = size
    ink_top, ink_bottom = mark_ink_box(s)
    dy = s / 2 - (ink_top + ink_bottom) / 2
    return (
        f'<g transform="translate(0,{dy:.3f})">'
        f'<path d="{chevron_path(s)}" fill="none" stroke="{AMBER}" '
        f'stroke-width="{s * 0.125:.2f}" stroke-linecap="butt" '
        f'stroke-linejoin="miter"/></g>'
    )


def wordmark_svg(x, y, size, fill, text="veloce", tracking="0.42"):
    """Lowercase wordmark with generous tracking: a compiler is quiet."""
    return (
        f'<text x="{x}" y="{y}" font-family="DejaVu Sans Mono, Liberation Mono, '
        f'monospace" font-size="{size}" font-weight="500" '
        f'letter-spacing="{size * float(tracking):.1f}" '
        f'fill="{fill}">{text}</text>'
    )


def svg_doc(w, h, body, bg=None):
    back = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="Veloce">\n'
        f'  {back}\n  {body}\n</svg>\n'
    )


# --- primary lockup: mark + wordmark ------------------------------------
def lockup(dark=True):
    """Mark on the left, wordmark right, tagline under it.

    The mark is vertically centred on the *wordmark's* optical centre rather
    than on the canvas: the tagline hangs below the word, so centring the mark
    on the canvas left it floating too high.
    """
    W, H = 660, 200
    bg = BG_DARK if dark else "#ffffff"
    ink = INK_ON_DARK if dark else INK_ON_LIGHT
    mark_box = 116
    # Text ink span, measured in the browser from the rendered SVG: the
    # wordmark occupies y=54..119 and the tagline y=127..143. Centre the mark
    # on that whole span -- centring it on the word alone left it top-heavy.
    text_top, text_bottom = 54.0, 143.0
    mark_y = (text_top + text_bottom) / 2 - mark_box / 2
    body = (
        f'<g transform="translate(38,{mark_y})">{mark_svg(mark_box)}</g>\n'
        f'  {wordmark_svg(180, 106, 56, ink)}\n'
        f'  <text x="183" y="140" font-family="DejaVu Sans Mono, monospace" '
        f'font-size="14.5" letter-spacing="3.4" '
        f'fill="{MUTED if dark else MUTED_LIGHT}">'
        f'LOW-BIT AOT COMPILER</text>' 
    )
    return svg_doc(W, H, body, bg)


def mark_only(dark=True, size=256):
    bg = BG_DARK if dark else "#ffffff"
    return svg_doc(size, size, mark_svg(size), bg)


# --- minimal PNG writer (stdlib only: no Pillow in this env) -----------
def _png(path, w, h, pixels):
    """pixels: list of rows, each a list of (r,g,b) tuples."""
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw += bytes((r, g, b))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _seg_dist(px, py, ax, ay, bx, by):
    """Distance from a point to a line segment (used to rasterise a stroke)."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    L2 = vx * vx + vy * vy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / L2))
    dx, dy = px - (ax + t * vx), py - (ay + t * vy)
    return (dx * dx + dy * dy) ** 0.5


def _render_mark_raw(size, dark=True, ss=3):
    """Supersampled pixels for the chevron mark, without any imaging dependency.

    Shapes are evaluated analytically at supersampled resolution and box
    filtered down, which gives clean anti-aliased edges without needing
    Pillow/cairo (neither is available here).
    """
    bg = hex_rgb(BG_DARK if dark else "#ffffff")
    amber = hex_rgb(AMBER)
    W = size * ss

    # --- one V stroke, opaque, in supersampled units ---
    pad = W * 0.075
    top = W * V_TOP
    bottom = W * V_BOTTOM
    cx = W / 2
    vy = top + (bottom - top) * V_DEPTH
    ink_top, ink_bottom = mark_ink_box(W)
    dy = W / 2 - (ink_top + ink_bottom) / 2
    top += dy; vy += dy
    # one entry per segment: (ax, ay, bx, by, half_width)
    strokes = [
        (pad, top, cx, vy, W * 0.125 / 2),
        (cx, vy, W - pad, top, W * 0.125 / 2),
    ]


    img = []
    for y in range(W):
        row = []
        for x in range(W):
            c = list(bg)
            for (ax_, ay_, bx_, by_, half) in strokes:
                if _seg_dist(x, y, ax_, ay_, bx_, by_) <= half:
                    c = list(amber)
                    break
            row.append(tuple(c))
        img.append(row)

    return img, bg, ss


def _render_mark_pixels(size, dark=True, ss=4):
    """Down-filtered RGB rows for the mark (used by the PNG and ICO writers)."""
    img, bg, ss = _render_mark_raw(size, dark, ss)
    out = []
    for oy in range(size):
        row = []
        for ox in range(size):
            rs = gs = bs = 0
            for dy in range(ss):
                base = img[oy * ss + dy]
                for dx in range(ss):
                    c = base[ox * ss + dx]
                    rs += c[0]; gs += c[1]; bs += c[2]
            m = ss * ss
            row.append((rs // m, gs // m, bs // m))
        out.append(row)
    return out


def render_mark_png(path, size=1024, dark=True, ss=3):
    """Rasterise the mark to a PNG."""
    _png(path, size, size, _render_mark_pixels(size, dark, ss))
    return path


def _ico(path, sizes=(16, 24, 32, 48, 64)):
    """Write a multi-resolution .ico (uncompressed 32-bit BGRA entries).

    Stdlib only: ICO with BMP payloads is the most compatible form and needs no
    PNG encoder tricks -- just a header, a directory, and the pixel data read
    back out of our own supersampled rasteriser.
    """
    entries = []
    for n in sizes:
        px = _render_mark_pixels(n, dark=True)
        # BMP-in-ICO: bottom-up rows, 32bpp BGRA, plus a 1bpp AND mask
        hdr = struct.pack("<IiiHHIIiiII", 40, n, n * 2, 1, 32, 0, 0, 0, 0, 0, 0)
        rows = bytearray()
        for y in range(n - 1, -1, -1):
            for x in range(n):
                r, g, b = px[y][x]
                rows += bytes((b, g, r, 255))
        mask_row = ((n + 31) // 32) * 4
        rows += b"\x00" * (mask_row * n)
        entries.append((n, hdr + bytes(rows)))

    out = bytearray(struct.pack("<HHH", 0, 1, len(entries)))
    offset = 6 + 16 * len(entries)
    for n, data in entries:
        out += struct.pack("<BBBBHHII", n if n < 256 else 0, n if n < 256 else 0,
                           0, 0, 1, 32, len(data), offset)
        offset += len(data)
    for _, data in entries:
        out += data
    with open(path, "wb") as f:
        f.write(out)
    return path


def main():
    files = {
        "logo.svg": lockup(dark=True),
        "logo-light.svg": lockup(dark=False),
        "logo-mark.svg": mark_only(dark=True),
        "logo-mark-light.svg": mark_only(dark=False),
        "logo-wordmark.svg": svg_doc(
            420, 140,
            wordmark_svg(10, 92, 64, INK_ON_LIGHT), "#ffffff"),
    }
    for name, text in files.items():
        with open(os.path.join(HERE, name), "w") as f:
            f.write(text)
        print("wrote", name)
    print("wrote", render_mark_png(os.path.join(HERE, "logo.png"), 1024, True))
    print("wrote", render_mark_png(os.path.join(HERE, "logo-256.png"), 256, True))
    print("wrote", _ico(os.path.join(HERE, "favicon.ico")))


if __name__ == "__main__":
    main()
