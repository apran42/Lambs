"""TensorRT 8.x engine runner that does not require PyCUDA."""

import time

import numpy as np

from jetson_worker.cuda_runtime import CudaRuntime


class TensorRTEngine(object):
    def __init__(self, engine_path):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT is unavailable. Run this module with Jetson system Python 3.6."
            ) from exc

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as engine_file:
            serialized = engine_file.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError("TensorRT could not deserialize {}".format(engine_path))
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")

        self.cuda = CudaRuntime()
        self.device_buffers = []
        self.bindings = [0] * self.engine.num_bindings
        self.input_index = None
        self.output_indices = []
        self.binding_shapes = {}
        self.binding_dtypes = {}

        for index in range(self.engine.num_bindings):
            shape = tuple(int(value) for value in self.engine.get_binding_shape(index))
            if any(value < 0 for value in shape):
                raise RuntimeError(
                    "Dynamic TensorRT bindings are not supported yet: {}".format(shape)
                )
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(index)))
            item_count = int(trt.volume(shape))
            if self.engine.has_implicit_batch_dimension:
                item_count *= int(self.engine.max_batch_size)
            device = self.cuda.malloc(item_count * dtype.itemsize)
            self.device_buffers.append(device)
            self.bindings[index] = int(device.value)
            self.binding_shapes[index] = shape
            self.binding_dtypes[index] = dtype
            if self.engine.binding_is_input(index):
                if self.input_index is not None:
                    raise RuntimeError("The first worker supports exactly one input")
                self.input_index = index
            else:
                self.output_indices.append(index)

        if self.input_index is None or not self.output_indices:
            raise RuntimeError("TensorRT engine must have one input and at least one output")

    @property
    def input_shape(self):
        return self.binding_shapes[self.input_index]

    def infer(self, input_tensor):
        expected_shape = self.input_shape
        if tuple(input_tensor.shape) != expected_shape:
            raise ValueError(
                "Input shape {} does not match engine shape {}".format(
                    input_tensor.shape,
                    expected_shape,
                )
            )
        expected_dtype = self.binding_dtypes[self.input_index]
        input_tensor = np.ascontiguousarray(input_tensor, dtype=expected_dtype)
        outputs = [
            np.empty(
                self.binding_shapes[index],
                dtype=self.binding_dtypes[index],
            )
            for index in self.output_indices
        ]

        started = time.time()
        self.cuda.copy_host_to_device(
            self.device_buffers[self.input_index],
            input_tensor,
        )
        succeeded = self.context.execute_v2(self.bindings)
        if not succeeded:
            raise RuntimeError("TensorRT execute_v2 returned false")
        for output, index in zip(outputs, self.output_indices):
            self.cuda.copy_device_to_host(output, self.device_buffers[index])
        elapsed_ms = (time.time() - started) * 1000.0
        return outputs, elapsed_ms

    def close(self):
        if getattr(self, "device_buffers", None):
            for pointer in self.device_buffers:
                self.cuda.free(pointer)
            self.device_buffers = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

