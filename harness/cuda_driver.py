"""
cuda_driver.py — Thin ctypes wrapper around the CUDA Driver API.

No ptxas, no pycuda, no CUDA toolkit needed — only libcuda.so (CUDA driver).
The driver JIT-compiles PTX to native CUBIN internally when the module is loaded.

Public API:
    init()                            — initialise driver + create context
    load_ptx(path) -> module          — load a PTX file, get CUmodule
    get_function(module, name) -> fn  — get CUfunction by name
    alloc(n_bytes) -> DevicePtr       — allocate device memory
    memcpy_h2d(dst, src_np)           — host numpy → device
    memcpy_d2h(dst_np, src)           — device → host numpy
    memset(dst, value, n_bytes)       — fill device memory with a byte value
    launch(fn, grid, block, *args)    — launch kernel
    synchronize()                     — wait for all work to complete
    free(ptr)                         — free device memory
    event_create() -> CUevent         — allocate a CUDA event
    event_record(event)               — record event on default stream
    event_elapsed_ms(start, stop)     — kernel-only elapsed time in ms
    event_destroy(event)              — release a CUDA event
"""

import ctypes
import ctypes.util
import numpy as np
from typing import Any, Tuple

# ── Load the CUDA driver library ──────────────────────────────────────────────
_lib_path = ctypes.util.find_library("cuda") or "libcuda.so"
_cuda = ctypes.CDLL(_lib_path, use_errno=True)

# C types shortcuts
_p = ctypes.POINTER
_u64 = ctypes.c_uint64
_i64 = ctypes.c_int64
_i32 = ctypes.c_int32
_vp  = ctypes.c_void_p
_cp  = ctypes.c_char_p
_sz  = ctypes.c_size_t

# CUresult error check
def _chk(r, *_):
    if r != 0:
        name = ctypes.create_string_buffer(256)
        _cuda.cuGetErrorName(r, ctypes.byref(ctypes.cast(name, _p(_cp))))
        desc = ctypes.create_string_buffer(256)
        _cuda.cuGetErrorString(r, ctypes.byref(ctypes.cast(desc, _p(_cp))))
        raise RuntimeError(
            f"CUDA Driver error {r}: {name.value.decode()} — {desc.value.decode()}"
        )

# Wrap each API call with errcheck
def _fn(name, restype, *argtypes):
    fn = getattr(_cuda, name)
    fn.restype  = restype
    fn.argtypes = argtypes
    fn.errcheck = _chk
    return fn

_cuInit              = _fn("cuInit",              _i32, _i32)
_cuDeviceGet         = _fn("cuDeviceGet",         _i32, _p(_i32), _i32)
_cuCtxCreate         = _fn("cuCtxCreate_v2",      _i32, _p(_vp), _i32, _i32)
_cuModuleLoadData    = _fn("cuModuleLoadData",    _i32, _p(_vp), _cp)
_cuModuleGetFunction = _fn("cuModuleGetFunction", _i32, _p(_vp), _vp, _cp)
_cuMemAlloc          = _fn("cuMemAlloc_v2",       _i32, _p(_u64), _sz)
_cuMemcpyHtoD        = _fn("cuMemcpyHtoD_v2",    _i32, _u64, _vp, _sz)
_cuMemcpyDtoH        = _fn("cuMemcpyDtoH_v2",    _i32, _vp, _u64, _sz)
_cuLaunchKernel      = _fn("cuLaunchKernel",      _i32,
                            _vp,                  # CUfunction
                            _i32, _i32, _i32,     # gridX, gridY, gridZ
                            _i32, _i32, _i32,     # blockX, blockY, blockZ
                            _i32,                 # sharedMemBytes
                            _vp,                  # hStream
                            _p(_vp),              # kernelParams (void**)
                            _p(_vp))              # extra
_cuCtxSynchronize    = _fn("cuCtxSynchronize",    _i32)
_cuMemFree           = _fn("cuMemFree_v2",        _i32, _u64)
_cuMemsetD8          = _fn("cuMemsetD8_v2",       _i32, _u64, ctypes.c_ubyte, _sz)
_cuEventCreate       = _fn("cuEventCreate",       _i32, _p(_vp), _i32)
_cuEventRecord       = _fn("cuEventRecord",       _i32, _vp, _vp)
_cuEventSynchronize  = _fn("cuEventSynchronize",  _i32, _vp)
_cuEventElapsedTime  = _fn("cuEventElapsedTime",  _i32, _p(ctypes.c_float), _vp, _vp)
_cuEventDestroy      = _fn("cuEventDestroy",      _i32, _vp)

# ── High-level helpers ─────────────────────────────────────────────────────────

class DevicePtr:
    """Wraps a CUdeviceptr (uint64 device address)."""
    def __init__(self, addr: int, nbytes: int):
        self._addr   = _u64(addr)
        self.nbytes  = nbytes

    @property
    def addr(self) -> _u64:
        return self._addr

    @property
    def addr_int(self) -> int:
        return self._addr.value

    def free(self):
        _cuMemFree(self._addr)

    def __repr__(self):
        return f"DevicePtr(0x{self._addr.value:016x}, {self.nbytes}B)"


_ctx = None   # global CUDA context (one per process is enough)

def init():
    """Initialise the CUDA driver and create a context on device 0."""
    global _ctx
    if _ctx is not None:
        return
    _cuInit(0)
    dev = _i32(0)
    _cuDeviceGet(ctypes.byref(dev), 0)
    ctx = _vp()
    _cuCtxCreate(ctypes.byref(ctx), 0, dev)
    _ctx = ctx


def load_ptx(ptx_path: str):
    """
    Load a PTX file.  The driver JIT-compiles it to native code.
    Returns a CUmodule (ctypes void_p).
    """
    with open(ptx_path, "rb") as f:
        ptx_bytes = f.read()
    module = _vp()
    _cuModuleLoadData(ctypes.byref(module), ptx_bytes)
    return module


def get_function(module, name: str):
    """Return a CUfunction (ctypes void_p) for the given kernel name."""
    fn = _vp()
    _cuModuleGetFunction(ctypes.byref(fn), module, name.encode())
    return fn


def alloc(n_bytes: int) -> DevicePtr:
    """Allocate n_bytes on the device.  Returns a DevicePtr."""
    addr = _u64(0)
    _cuMemAlloc(ctypes.byref(addr), n_bytes)
    return DevicePtr(addr.value, n_bytes)


def memcpy_h2d(dst: DevicePtr, src: np.ndarray):
    """Copy a contiguous numpy array to device memory."""
    src_c = np.ascontiguousarray(src)
    _cuMemcpyHtoD(dst.addr, src_c.ctypes.data_as(_vp), src_c.nbytes)


def memcpy_d2h(dst: np.ndarray, src: DevicePtr):
    """Copy device memory into an existing numpy array (must be contiguous)."""
    assert dst.nbytes == src.nbytes, (
        f"size mismatch: dst={dst.nbytes}, src={src.nbytes}")
    _cuMemcpyDtoH(dst.ctypes.data_as(_vp), src.addr, src.nbytes)


def launch(fn, grid: Tuple[int,int,int], block: Tuple[int,int,int],
           *args, shared_bytes: int = 0):
    """
    Launch a CUDA kernel.

    Each element of *args must be one of:
      int / np.int64    → passed as i64 parameter
      np.float32        → passed as f32 parameter
      DevicePtr         → passed as u64 device address

    The MLIR memref descriptor ABI requires that device pointers
    (base_ptr and aligned_ptr) be passed as DevicePtr, and all
    scalar fields (offset, size, stride) as int / np.int64.
    """
    c_args = []   # keep ctypes values alive
    ptrs   = []   # void* pointers into c_args

    for a in args:
        if isinstance(a, DevicePtr):
            c = _u64(a.addr_int)
        elif isinstance(a, (int, np.integer)):
            c = _i64(int(a))
        elif isinstance(a, float):
            c = ctypes.c_float(a)
        else:
            raise TypeError(f"unsupported kernel arg type: {type(a)}")
        c_args.append(c)
        ptrs.append(ctypes.cast(ctypes.byref(c), _vp))

    param_arr = (_vp * len(ptrs))(*ptrs)
    _cuLaunchKernel(
        fn,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        shared_bytes,
        None,        # stream: use default
        param_arr,
        None,        # extra: not used
    )


def memset(dst: "DevicePtr", value: int, n_bytes: int):
    """Fill the first n_bytes of dst with the byte value (0..255)."""
    _cuMemsetD8(dst.addr, value, n_bytes)


def synchronize():
    """Block until all work on the current context finishes."""
    _cuCtxSynchronize()


# ── CUDA event API ─────────────────────────────────────────────────────────────
# Events measure kernel-only elapsed time, excluding Python/driver call overhead.

_CU_EVENT_DEFAULT = 0

def event_create():
    """Allocate and return a new CUevent (ctypes void_p)."""
    h = _vp()
    _cuEventCreate(ctypes.byref(h), _CU_EVENT_DEFAULT)
    return h


def event_record(event):
    """Enqueue a timestamp for *event* on the default (null) stream."""
    _cuEventRecord(event, _vp())   # null stream


def event_elapsed_ms(start, stop) -> float:
    """
    Return the elapsed time in milliseconds between two previously recorded
    events.  Synchronises on *stop* to ensure it has completed.
    """
    _cuEventSynchronize(stop)
    ms = ctypes.c_float(0.0)
    _cuEventElapsedTime(ctypes.byref(ms), start, stop)
    return float(ms.value)


def event_destroy(event):
    """Release a CUevent."""
    _cuEventDestroy(event)
