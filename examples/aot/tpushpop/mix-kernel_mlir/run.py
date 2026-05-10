import ctypes
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401

from ptodsl.utils.npu_info import get_num_cube_cores, get_test_device

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "build_artifacts", "tpushpop_mlir_lib.so")
MODES = ("c2v", "c2v_add", "v2c", "bidi", "multi")
TILE = 16
FIFO_BYTES = 8 * 1024


def ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(LIB)
    lib.call_kernel.argtypes = [ctypes.c_uint32] + [ctypes.c_void_p] * 4
    return lib


def expected(mode: str, x: torch.Tensor) -> torch.Tensor:
    y = x.cpu() if mode == "v2c" else x.cpu() @ x.cpu()
    return 2 * y if mode in ("c2v_add", "bidi", "multi") else y


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "c2v"
    assert mode in MODES, f"Available modes: {MODES}"

    env = dict(os.environ, TPUSHPOP_MODE=mode)
    subprocess.run(["bash", "compile.sh"], check=True, cwd=HERE, env=env)

    device = get_test_device()
    torch.npu.set_device(device)

    blocks = get_num_cube_cores()
    shape = (blocks, TILE, TILE)
    fifo_bytes = 2 * FIFO_BYTES if mode == "multi" else FIFO_BYTES
    slots = torch.zeros((blocks * fifo_bytes,), dtype=torch.uint8, device=device)
    x = torch.rand(shape, dtype=torch.float32, device=device) - 0.5
    y = torch.zeros_like(x)

    load_lib().call_kernel(
        blocks,
        torch.npu.current_stream()._as_parameter_,
        ptr(slots),
        ptr(x),
        ptr(y),
    )
    torch.npu.synchronize()

    ref = expected(mode, x)
    assert torch.allclose(y.cpu(), ref, atol=1e-4, rtol=1e-4)
    max_abs = torch.max(torch.abs(y.cpu() - ref)).item()
    print(f"{mode} passed: shape={shape} max_abs={max_abs:.6f}")


if __name__ == "__main__":
    main()
