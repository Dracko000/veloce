"""Veloce weight-file (.veloce) serialisation — zero-copy model loading.

On-disk layout (FR-3.3) matches the compiler's packed in-memory layout, so a
mmap'd file can be handed straight to the kernels with no deserialisation:

    offset 0   : magic  b"VCWGT1\\x00\\x00"   (8 bytes)
    offset 8   : uint32 little-endian manifest length L
    offset 12  : L bytes of UTF-8 JSON manifest
    ...        : 64-byte aligned tensor blobs

manifest = {"align": 64,
            "tensors": [{"name", "dtype", "dims", "offset", "nbytes"}, ...],
            "model_card": {...}}   # optional; see veloce/capabilities.py

dtype layout:
    trit : flat base-3 packed, 5 trits per byte  ((numel + 4) // 5 bytes)
    i4   : flat 4-bit packed, 2 signed nibbles per byte ((numel + 1) // 2)
    fp16 : uint16 little-endian
    f32  : float32 little-endian
"""
import json
import struct

MAGIC = b"VCWGT1\x00\x00"
ALIGN = 64


def _as_int64_flat(values):
    """Return a flat int64 NumPy view/copy of ``values`` (list or array)."""
    import numpy as np
    if isinstance(values, np.ndarray):
        return values.reshape(-1).astype(np.int64, copy=False)
    return np.asarray(values, dtype=np.int64).reshape(-1)


def pack_trits(values):
    """Pack int values in {-1,0,1} into base-3 bytes, 5 per byte (flat).

    NumPy fast path: a 275M-trit model packs in ~1 s instead of minutes.
    """
    import numpy as np
    flat = _as_int64_flat(values)
    n = flat.size
    pad = (-n) % 5
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int64)])
    v = flat + 1
    if n and (int(v.min()) < 0 or int(v.max()) > 2):
        bad = int(flat[(v < 0) | (v > 2)][0])
        raise ValueError(f"trit value out of range: {bad}")
    weights = np.array([1, 3, 9, 27, 81], dtype=np.int64)
    packed = v.reshape(-1, 5) @ weights        # max 242, fits in uint8
    return packed.astype(np.uint8).tobytes()


def pack_i4(values):
    """Pack signed ints in [-8, 7] into bytes, 2 nibbles per byte (low first).

    NumPy fast path keeps multi-hundred-million-parameter models practical.
    """
    import numpy as np
    flat = _as_int64_flat(values)
    n = flat.size
    if n and (int(flat.min()) < -8 or int(flat.max()) > 7):
        bad = int(flat[(flat < -8) | (flat > 7)][0])
        raise ValueError(f"i4 value out of range [-8, 7]: {bad}")
    if n % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.int64)])
    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] & 0x0F) << 4
    return (lo | hi).astype(np.uint8).tobytes()

def unpack_i4(data, n):
    """Inverse of pack_i4 (for tooling/tests)."""
    out = []
    for i in range(n):
        b = data[i // 2]
        v = (b & 0x0F) if i % 2 == 0 else ((b >> 4) & 0x0F)
        out.append(v - 16 if v >= 8 else v)
    return out

def unpack_trits(data, n):
    """Inverse of pack_trits (for tooling/tests)."""
    return unpack_trits_array(data, n).tolist()


def unpack_trits_array(data, n):
    """Vectorised inverse of pack_trits; returns an int8 NumPy array."""
    import numpy as np
    buf = np.frombuffer(data, dtype=np.uint8, count=(n + 4) // 5)
    if buf.size == 0:
        return np.zeros(0, dtype=np.int8)
    vals = np.empty((buf.size, 5), dtype=np.int8)
    for k in range(5):
        # uint8 arithmetic: packed bytes are < 3**5 = 243
        vals[:, k] = ((buf // (3 ** k)) % 3).astype(np.int8) - 1
    return vals.reshape(-1)[:n]


def _encode(dtype, array):
    import numpy as np
    arr = np.asarray(array)
    numel = int(arr.size)
    if dtype == "trit":
        return pack_trits(arr.reshape(-1)), numel
    if dtype == "i4":
        return pack_i4(arr.reshape(-1)), numel
    if dtype == "fp16":
        return arr.astype(np.float16).tobytes(), numel
    if dtype == "f32":
        return arr.astype(np.float32).tobytes(), numel
    raise ValueError(f"unsupported dtype {dtype!r}")


def write_veloce_model(path, tensors, model_card=None):
    """Write a .veloce weight file.

    `tensors` is a list of dicts: {"name", "dtype", "data"} where data is a
    numpy array (or nested list).  `model_card` is an optional dict (see
    :mod:`veloce.capabilities`) stored verbatim in the manifest, so every
    exported model carries its declared capability profile with it.  Returns
    the manifest dict.
    """
    if model_card is not None:
        from .capabilities import validate_model_card
        validate_model_card(model_card)
    blobs = []
    entries = []
    # First lay out the manifest region size with placeholder offsets.
    for t in tensors:
        raw, numel = _encode(t["dtype"], t["data"])
        shape = list(getattr(t["data"], "shape", [])) or [numel]
        blobs.append(raw)
        entries.append({
            "name": t["name"], "dtype": t["dtype"], "dims": shape,
            "numel": numel, "nbytes": len(raw), "offset": 0,
        })
    # Compute offsets: start after header, each blob 64-byte aligned.
    def layout(manifest_len):
        base = 12 + manifest_len
        off = (base + ALIGN - 1) // ALIGN * ALIGN
        for e in entries:
            e["offset"] = off
            off = (off + e["nbytes"] + ALIGN - 1) // ALIGN * ALIGN
        return off

    # Manifest length depends on the (offset) numbers; iterate to a fixed point.
    man = {"align": ALIGN, "tensors": entries}
    if model_card is not None:
        man["model_card"] = model_card
    for _ in range(4):
        mjson = json.dumps(man).encode("utf-8")
        total = layout(len(mjson))
        man["total_bytes"] = total
        newjson = json.dumps(man).encode("utf-8")
        if newjson == mjson:
            break
        mjson = newjson
    mjson = json.dumps(man).encode("utf-8")
    total = layout(len(mjson))
    man["total_bytes"] = total
    mjson = json.dumps(man).encode("utf-8")

    buf = bytearray(total)
    buf[0:8] = MAGIC
    struct.pack_into("<I", buf, 8, len(mjson))
    buf[12:12 + len(mjson)] = mjson
    for e, raw in zip(entries, blobs):
        buf[e["offset"]:e["offset"] + len(raw)] = raw
    with open(path, "wb") as f:
        f.write(buf)
    return man


def read_veloce_manifest(path):
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:8] != MAGIC:
            raise ValueError("not a .veloce weight file")
        (mlen,) = struct.unpack("<I", head[8:12])
        man = json.loads(f.read(mlen).decode("utf-8"))
    return man


def save_veloce(path, tensors, model_card=None):
    """Write a ``.veloce`` file from a list of ``(name, array, dtype)`` tuples.

    Thin convenience wrapper around :func:`write_veloce_model`.  Pass
    ``model_card=`` (see :mod:`veloce.capabilities`) to embed the capability
    profile in the file.  Returns the total file size in bytes (each tensor
    blob is 64-byte aligned so the file is zero-copy mmap-ready, FR-3.3).
    """
    man = write_veloce_model(
        path,
        ({"name": n, "dtype": d, "data": a} for (n, a, d) in tensors),
        model_card=model_card)
    return man["total_bytes"]


class VeloceFile:
    """Read-only view over a ``.veloce`` model, backed by a zero-copy mmap.

    Attributes
    ----------
    entries : dict mapping tensor name -> manifest entry
              (keys: ``name``, ``dtype``, ``dims``, ``numel``, ``nbytes``,
              ``offset``).
    model_card : the embedded capability model card (dict) or ``None``.
    names()  : tensor names in file order.
    numpy(n) : decoded NumPy array for tensor ``n`` (a copy).
    raw(n)   : zero-copy ``memoryview`` of tensor ``n``'s packed bytes
               (trit blobs stay packed exactly as the runtime expects).
    """

    def __init__(self, path):
        import mmap
        manifest = read_veloce_manifest(path)
        self.path = path
        self.align = manifest.get("align", ALIGN)
        self.entries = {e["name"]: e for e in manifest["tensors"]}
        self.model_card = manifest.get("model_card")
        self._order = [e["name"] for e in manifest["tensors"]]
        self._f = open(path, "rb")
        try:
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):  # empty/unsupported -> byte buffer
            self._mm = self._f.read()
        self._closed = False

    def names(self):
        return list(self._order)

    def _blob(self, name):
        e = self.entries[name]
        return self._mm[e["offset"]:e["offset"] + e["nbytes"]]

    def raw(self, name):
        """Zero-copy view of the packed bytes of tensor ``name``."""
        e = self.entries[name]
        return memoryview(self._mm)[e["offset"]:e["offset"] + e["nbytes"]]

    def numpy(self, name):
        """Decode tensor ``name`` into a NumPy array (copy)."""
        import numpy as np
        e = self.entries[name]
        blob = self._blob(name)
        dims = tuple(e.get("dims") or (e["numel"],))
        dt = e["dtype"]
        if dt == "trit":
            flat = unpack_trits_array(blob, e["numel"])
            return flat.reshape(dims)
        if dt == "i4":
            flat = unpack_i4(blob, e["numel"])
            return np.array(flat, dtype=np.int8).reshape(dims)
        if dt == "fp16":
            return np.frombuffer(blob, dtype="<f2").reshape(dims)
        if dt == "f32":
            return np.frombuffer(blob, dtype="<f4").reshape(dims)
        raise ValueError(f"unsupported dtype {dt!r}")

    def close(self):
        if self._closed:
            return
        try:
            if hasattr(self._mm, "close"):
                self._mm.close()
        finally:
            self._f.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
