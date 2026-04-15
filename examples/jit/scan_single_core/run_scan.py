import torch
import torch_npu
from ptodsl import jit, pto, tile
from ptodsl import scalar as s
from ptodsl.test_util import get_test_device

const = s.const

def meta_data():
    dtype = pto.float32
    ptr_type = pto.PtrType(dtype)
    index_dtype = pto.int32

    tensor_type = pto.TensorType(rank=1, dtype=dtype)
    tensor_type_2d = pto.TensorType(rank=2, dtype=dtype)

    subtensor_type_u = pto.SubTensorType(shape=[64, 64], dtype=dtype)
    subtensor_type_a = pto.SubTensorType(shape=[64, 64], dtype=dtype)
    subtensor_type_c = pto.SubTensorType(shape=[64, 64], dtype=dtype)
    subtensor_type_row = pto.SubTensorType(shape=[1, 64], dtype=dtype)

    tile_cfg_mat_a = pto.TileBufConfig(blayout="ColMajor", slayout="RowMajor")
    tile_cfg_mat_u = pto.TileBufConfig(blayout="RowMajor", slayout="ColMajor")
    tile_cfg_acc = pto.TileBufConfig(blayout="ColMajor", slayout="RowMajor")

    tile_type_a_l1 = pto.TileBufType(shape=[64, 64], dtype=dtype, memory_space="MAT", config=tile_cfg_mat_a)
    tile_type_u_l1 = pto.TileBufType(shape=[64, 64], dtype=dtype, memory_space="MAT", config=tile_cfg_mat_u)

    tile_type_a = pto.TileBufType(shape=[64, 64], dtype=dtype, memory_space="LEFT", config=tile_cfg_mat_a)
    tile_type_u = pto.TileBufType(shape=[64, 64], dtype=dtype, memory_space="RIGHT", config=tile_cfg_mat_u)
    tile_type_c = pto.TileBufType(shape=[64, 64], dtype=dtype, memory_space="ACC", config=tile_cfg_acc)

    tile_type_row = pto.TileBufType(shape=[1, 64], valid_shape=[1, 64], dtype=dtype, memory_space="VEC", config=pto.TileBufConfig())
    tile_type_1x1 = pto.TileBufType(shape=[1, 1], valid_shape=[1, 1], dtype=dtype, memory_space="VEC", config=pto.TileBufConfig())

    return {
        "ptr_type": ptr_type,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "tensor_type_2d": tensor_type_2d,
        "subtensor_type_u": subtensor_type_u,
        "subtensor_type_a": subtensor_type_a,
        "subtensor_type_c": subtensor_type_c,
        "subtensor_type_row": subtensor_type_row,
        "tile_type_a_l1": tile_type_a_l1,
        "tile_type_u_l1": tile_type_u_l1,
        "tile_type_a": tile_type_a,
        "tile_type_u": tile_type_u,
        "tile_type_c": tile_type_c,
        "tile_type_row": tile_type_row,
        "tile_type_1x1": tile_type_1x1,
    }

@jit(meta_data=meta_data, block_dim=1, enable_insert_sync=True)
def run_scan_kernel(
    x_ptr: "ptr_type",
    out_ptr: "ptr_type",
    u_ptr: "ptr_type",
    total_len_i32: "index_dtype",
) -> None:
    c0 = const(0)
    c1 = const(1)
    c63 = const(63)
    c64 = const(64)
    c4096 = const(4096)

    total_len = s.index_cast(total_len_i32)
    num_chunks = total_len // c4096

    with pto.cube_section():
        tvU = pto.as_tensor(tensor_type_2d, ptr=u_ptr, shape=[c64, c64], strides=[c1, c64], layout="DN")
        tvX_cube = pto.as_tensor(tensor_type_2d, ptr=x_ptr, shape=[total_len // c64, c64], strides=[c64, c1])
        tvOut_cube = pto.as_tensor(tensor_type_2d, ptr=out_ptr, shape=[total_len // c64, c64], strides=[c64, c1])

        uTileL1 = pto.alloc_tile(tile_type_u_l1)
        uTile = pto.alloc_tile(tile_type_u)
        aTileL1 = pto.alloc_tile(tile_type_a_l1)
        aTile = pto.alloc_tile(tile_type_a)
        cTile = pto.alloc_tile(tile_type_c)

        svU = pto.slice_view(subtensor_type_u, source=tvU, offsets=[c0, c0], sizes=[c64, c64])
        pto.load(svU, uTileL1)
        tile.mov(uTileL1, uTile)

        for chunk in pto.range(c0, num_chunks, c1):
            offset = chunk * c64
            svX = pto.slice_view(subtensor_type_a, source=tvX_cube, offsets=[offset, c0], sizes=[c64, c64])
            svOut = pto.slice_view(subtensor_type_c, source=tvOut_cube, offsets=[offset, c0], sizes=[c64, c64])

            pto.load(svX, aTileL1)
            tile.mov(aTileL1, aTile)
            tile.matmul(aTile, uTile, cTile)
            pto.store(cTile, svOut)

    with pto.vector_section():
        tvOut_vec = pto.as_tensor(tensor_type_2d, ptr=out_ptr, shape=[total_len // c64, c64], strides=[c64, c1])

        rowTile = pto.alloc_tile(tile_type_row)
        sumTile1x1 = pto.alloc_tile(tile_type_1x1)
        sumTileExpanded = pto.alloc_tile(tile_type_row)

        for chunk in pto.range(c0, num_chunks, c1):
            chunk_offset = chunk * c64
            for r in pto.range(c0, c64, c1):
                offset = chunk_offset + r
                svRow = pto.slice_view(subtensor_type_row, source=tvOut_vec, offsets=[offset, c0], sizes=[c1, c64])

                pto.load(svRow, rowTile)

                is_first = s.eq(offset, c0)

                def handle_add():
                    tile.col_expand(sumTile1x1, sumTileExpanded)
                    tile.add(rowTile, sumTileExpanded, rowTile)

                with pto.if_context(is_first, has_else=True) as branch:
                    pass
                with branch.else_context():
                    handle_add()

                tile.extract(rowTile, c0, c63, sumTile1x1)
                pto.store(rowTile, svRow)

def test_scan():
    device = get_test_device()
    torch.npu.set_device(device)

    total_len = 4096 * 4
    torch.manual_seed(0)
    dtype = torch.float32

    x = torch.rand(total_len, device=device, dtype=dtype)
    out = torch.empty(total_len, device=device, dtype=dtype)

    # U is an upper triangular matrix of 1s (to scan the chunk)
    u = torch.ones((64, 64), dtype=dtype, device=device)
    u = torch.triu(u)

    run_scan_kernel(x, out, u, total_len)
    torch.npu.synchronize()

    ref = torch.cumsum(x, dim=0)
    # The scan precision error might accumulate over 4 chunks, use appropriate rtol/atol
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("result equal!")

if __name__ == "__main__":
    test_scan()
