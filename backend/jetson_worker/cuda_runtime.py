"""Small CUDA Runtime wrapper used when PyCUDA is unavailable.

The Jetson Nano system image already provides ``libcudart`` through JetPack.
Keeping the wrapper here avoids modifying the system Python installation merely
to allocate TensorRT input and output buffers.
"""

import ctypes
import ctypes.util


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


class CudaRuntimeError(RuntimeError):
    pass


def _load_cudart():
    candidates = [
        ctypes.util.find_library("cudart"),
        "libcudart.so.10.2",
        "libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so.10.2",
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append("{}: {}".format(candidate, exc))
    raise CudaRuntimeError(
        "Could not load the CUDA runtime library. Tried: {}".format(
            "; ".join(errors) or "no library candidates"
        )
    )


class CudaRuntime(object):
    def __init__(self):
        self.lib = _load_cudart()
        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, code, operation):
        if code == 0:
            return
        message = self.lib.cudaGetErrorString(code)
        if message:
            message = message.decode("utf-8", errors="replace")
        else:
            message = "unknown CUDA error"
        raise CudaRuntimeError("{} failed: {} ({})".format(operation, message, code))

    def malloc(self, byte_count):
        pointer = ctypes.c_void_p()
        self._check(
            self.lib.cudaMalloc(ctypes.byref(pointer), int(byte_count)),
            "cudaMalloc",
        )
        return pointer

    def free(self, pointer):
        if pointer and pointer.value:
            self._check(self.lib.cudaFree(pointer), "cudaFree")
            pointer.value = None

    def copy_host_to_device(self, device_pointer, host_array):
        self._check(
            self.lib.cudaMemcpy(
                device_pointer,
                ctypes.c_void_p(host_array.ctypes.data),
                host_array.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy host-to-device",
        )

    def copy_device_to_host(self, host_array, device_pointer):
        self._check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(host_array.ctypes.data),
                device_pointer,
                host_array.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy device-to-host",
        )

