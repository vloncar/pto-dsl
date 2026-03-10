import ctypes
import os
import subprocess

import pytest
import torch

from ptodsl.test_util import get_test_device

torch.manual_seed(0)

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = get_test_device()
_ROWSUM_LIB_PATH = os.path.join(_DIR, "rowsum_lib.so")
_COLSUM_LIB_PATH = os.path.join(_DIR, "colsum_lib.so")
_BLOCK_DIM = 24

_SHAPES = [
    (1, 1),
    (2, 3),
    (3, 2),
    (7, 7),
    (15, 17),
    (17, 15),
    (31, 32),
    (32, 31),
    (32, 32),
    (33, 33),
    (31, 33),
    (33, 31),
    (64, 32),
    (32, 64),
    (63, 64),
    (64, 63),
    (65, 33),
    (65, 64),
    (29, 257),
    (127, 129),
]

_SHAPE_IDS = [f"batch{batch}-cols{n_cols}" for batch, n_cols in _SHAPES]


@pytest.fixture(scope="session")
def compiled_sum():
    subprocess.check_call(["bash", os.path.join(_DIR, "compile.sh")], cwd=_DIR)
    yield
    if os.path.exists(_ROWSUM_LIB_PATH):
        os.remove(_ROWSUM_LIB_PATH)
    if os.path.exists(_COLSUM_LIB_PATH):
        os.remove(_COLSUM_LIB_PATH)


def test_build_rowsum(compiled_sum):
    assert os.path.exists(_ROWSUM_LIB_PATH)


def test_build_colsum(compiled_sum):
    assert os.path.exists(_COLSUM_LIB_PATH)


@pytest.mark.require_npu
@pytest.mark.parametrize("batch, n_cols", _SHAPES, ids=_SHAPE_IDS)
def test_rowsum_precision(compiled_sum, batch, n_cols):
    import torch_npu  # noqa: F401

    lib = ctypes.CDLL(_ROWSUM_LIB_PATH)
    lib.call_rowsum.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.call_rowsum.restype = None

    torch.npu.set_device(_DEVICE)
    x = torch.randn(batch, n_cols, device=_DEVICE, dtype=torch.float32)
    y = torch.full((batch,), float("nan"), device=_DEVICE, dtype=torch.float32)

    y_ref = x.float().sum(dim=-1)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_rowsum(
        ctypes.c_uint32(_BLOCK_DIM),
        stream_ptr,
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_uint32(batch),
        ctypes.c_uint32(n_cols),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=0)


@pytest.mark.require_npu
@pytest.mark.parametrize("batch, n_cols", _SHAPES, ids=_SHAPE_IDS)
def test_colsum_precision(compiled_sum, batch, n_cols):
    import torch_npu  # noqa: F401

    lib = ctypes.CDLL(_COLSUM_LIB_PATH)
    lib.call_colsum.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.call_colsum.restype = None

    torch.npu.set_device(_DEVICE)
    x = torch.randn(batch, n_cols, device=_DEVICE, dtype=torch.float32)
    y = torch.full((n_cols,), float("nan"), device=_DEVICE, dtype=torch.float32)

    y_ref = x.float().sum(dim=0)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_colsum(
        ctypes.c_uint32(_BLOCK_DIM),
        stream_ptr,
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_uint32(batch),
        ctypes.c_uint32(n_cols),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])