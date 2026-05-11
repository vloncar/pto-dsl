"""Prefix-sum (scan) on a single AI Core — TPush/TPop synchronization.

This is a port of ``run_scan_single_core.py`` that replaces the legacy
``sync_set`` / ``sync_wait`` (TSync-based) Cube ↔ Vector handshake with the
structured TPush / TPop pipe primitives.

Algorithm overview (unchanged from the TSync version):
    1. Cube loads U (upper-triangular matrix of 1s) once.
    2. For each input tile X_i:
       a. Cube computes C_i = X_i @ U  (matmul → partial prefix sums).
       b. Cube pushes the ACC tile to Vector via the C2V pipe.
       c. Vector pops the tile, adds the running sum to each row,
          stores the result back to GM.
       d. Vector pushes a (dummy) signal tile back to Cube via V2C pipe
          so Cube knows the tile is done and can advance.

Synchronization:
    - C2V pipe (dir_mask=1): Cube pushes ACC tiles to Vector.
    - V2C pipe (dir_mask=2): Vector pushes a signal tile back to Cube.
    Both pipes use GM-staged L2G2L transport (A2/A3 path).

Usage:
    python run_scan_single_core_tpushpop.py
"""

import torch
import torch_npu
from ptodsl import jit, pto, tile
from ptodsl import scalar as s
from ptodsl.npu_info import get_test_device

TILE_SIZE = 64

const = s.const

# --- Pipe parameters ---
# C2V: Cube pushes TILE_SIZE x TILE_SIZE f32 ACC tiles to Vector.
# slot_size = TILE_SIZE * TILE_SIZE * sizeof(f32)
SLOT_SIZE_C2V = TILE_SIZE * TILE_SIZE * 4
# V2C: Vector pushes a tiny 1x8 f32 VEC signal tile back to Cube.
# We use a minimal 1x8 tile (32 bytes, but slot_size must cover the
# fractal footprint; for a 1×8 f32 VEC tile the minimum is 1*8*4=32 bytes,
# but pto-isa rounds up internally). We keep this at 32 bytes.
SLOT_SIZE_V2C = 1 * 8 * 4

# dir_mask=1 or 2 → slot_num=8 (§4.4 of the design doc).
SLOT_NUM = 8

# GM buffer sizes (in f32 elements): slot_size * slot_num / sizeof(f32)
GM_C2V_ELEMS = (SLOT_SIZE_C2V * SLOT_NUM) // 4
GM_V2C_ELEMS = (SLOT_SIZE_V2C * SLOT_NUM) // 4
GM_TOTAL_ELEMS = GM_C2V_ELEMS + GM_V2C_ELEMS

# reserve_buffer sizes (in bytes)
C2V_FIFO_BYTES = SLOT_SIZE_C2V * SLOT_NUM  # consumer-side VEC buffer
V2C_FIFO_BYTES = SLOT_SIZE_V2C * SLOT_NUM  # consumer-side MAT buffer

# Frontend pipe IDs (arbitrary, must be unique per-function).
ID_C2V = 0
ID_V2C = 1

SPLIT_NONE = 0


def meta_data():
    dtype = pto.float32
    ptr_type = pto.PtrType(dtype)
    ffts_type = pto.ffts_type
    len_type = pto.int32
    i32 = pto.int32

    tensor_type = pto.TensorType(rank=2, dtype=dtype)

    subtensor_type_u = pto.SubTensorType(shape=[TILE_SIZE, TILE_SIZE], dtype=dtype)
    subtensor_type_a = pto.SubTensorType(shape=[TILE_SIZE, TILE_SIZE], dtype=dtype)
    subtensor_type_c = pto.SubTensorType(shape=[TILE_SIZE, TILE_SIZE], dtype=dtype)
    subtensor_type_row = pto.SubTensorType(shape=[1, TILE_SIZE], dtype=dtype)

    tile_cfg_mat = pto.TileBufConfig(blayout="ColMajor", slayout="RowMajor")
    tile_cfg_left = pto.TileBufConfig(blayout="RowMajor", slayout="RowMajor")
    tile_cfg_right = pto.TileBufConfig(blayout="RowMajor", slayout="ColMajor")
    tile_cfg_acc = pto.TileBufConfig(blayout="ColMajor", slayout="RowMajor")

    tile_type_a_l1 = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="MAT",
        config=tile_cfg_mat,
    )
    tile_type_u_l1 = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="MAT",
        config=tile_cfg_mat,
    )

    tile_type_a = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="LEFT",
        config=tile_cfg_left,
    )
    tile_type_u = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="RIGHT",
        config=tile_cfg_right,
    )
    tile_type_c = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="ACC",
        config=tile_cfg_acc,
    )

    # Vector-side tile type for receiving the C2V accumulator pop.
    # The ACC tile from cube is popped as a VEC-space tile.
    tile_type_c_vec = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )

    tile_type_row = pto.TileBufType(
        shape=[1, TILE_SIZE],
        valid_shape=[1, TILE_SIZE],
        dtype=dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )
    tile_type_1x8 = pto.TileBufType(
        shape=[1, 8],
        valid_shape=[1, 8],
        dtype=dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )

    # Signal tile: Cube side receives into MAT space for V2C pop.
    tile_type_signal_mat = pto.TileBufType(
        shape=[1, 8],
        valid_shape=[1, 8],
        dtype=dtype,
        memory_space="MAT",
        config=pto.TileBufConfig(
            blayout="RowMajor",
            slayout="NoneBox",
            s_fractal_size=32,
        ),
    )

    return locals()


@jit(
    meta_data=meta_data,
    block_dim=1,
    module=True,
    enable_insert_sync=True,
    init_ffts="ffts_addr",
)
def scan_module():
    # ---------------------------------------------------------------
    # Cube kernel
    # ---------------------------------------------------------------
    @pto.func(kernel="cube")
    def cube_kernel(
        gm_slot_buffer: "ptr_type",
        x_ptr: "ptr_type",
        u_ptr: "ptr_type",
        total_len_i32: "len_type",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cTILE_SIZE = const(TILE_SIZE)
        cN_TILE_ELEM = const(TILE_SIZE * TILE_SIZE)
        c0_i32 = const(0, type=i32)

        total_len = s.index_cast(total_len_i32)
        num_tiles = total_len // cN_TILE_ELEM

        # --- GM slot buffer partitioning (per-block) ---
        # Single-core, so block_idx is always 0, but keep the pattern
        # for correctness in a multi-block future.
        gm_c2v = gm_slot_buffer
        gm_v2c = pto.add_ptr(gm_slot_buffer, const(GM_C2V_ELEMS))

        # --- Initialize C2V pipe (Cube is producer, dir_mask=1) ---
        c2v_import = pto.import_reserved_buffer(
            name="c2v_fifo", peer_func="@vector_kernel"
        )
        pto.aic_initialize_pipe(
            id=ID_C2V,
            dir_mask=1,
            slot_size=SLOT_SIZE_C2V,
            gm_slot_buffer=gm_c2v,
            c2v_consumer_buf=c2v_import,
            v2c_consumer_buf=c0_i32,
        )

        # --- Initialize V2C pipe (Cube is consumer, dir_mask=2) ---
        v2c_local = pto.reserve_buffer(
            name="v2c_fifo", size=V2C_FIFO_BYTES, location="MAT"
        )
        pto.aic_initialize_pipe(
            id=ID_V2C,
            dir_mask=2,
            slot_size=SLOT_SIZE_V2C,
            gm_slot_buffer=gm_v2c,
            c2v_consumer_buf=c0_i32,
            v2c_consumer_buf=v2c_local,
        )

        # --- Tile allocations ---
        uTileL1 = pto.alloc_tile(tile_type_u_l1)
        uTile = pto.alloc_tile(tile_type_u)
        aTileL1 = pto.alloc_tile(tile_type_a_l1)
        aTile = pto.alloc_tile(tile_type_a)
        cTile = pto.alloc_tile(tile_type_c)

        # --- Load U matrix once ---
        tvU = pto.as_tensor(
            tensor_type,
            ptr=u_ptr,
            shape=[cTILE_SIZE, cTILE_SIZE],
            strides=[cTILE_SIZE, cTILE_SIZE],
            layout="ND",
        )
        svU = pto.slice_view(
            subtensor_type_u,
            source=tvU,
            offsets=[c0, c0],
            sizes=[cTILE_SIZE, cTILE_SIZE],
        )
        pto.load(svU, uTileL1)
        tile.mov(uTileL1, uTile)

        # --- GM tensor views for input ---
        tvX = pto.as_tensor(
            tensor_type,
            ptr=x_ptr,
            shape=[cTILE_SIZE, cTILE_SIZE],
            strides=[cTILE_SIZE, cTILE_SIZE],
            layout="ND",
        )

        for tile_idx in pto.range(c0, num_tiles, c1):
            offset = tile_idx * cTILE_SIZE

            svX = pto.slice_view(
                subtensor_type_a,
                source=tvX,
                offsets=[offset, c0],
                sizes=[cTILE_SIZE, cTILE_SIZE],
            )

            pto.load(svX, aTileL1)

            tile.mov(aTileL1, aTile)

            tile.matmul(aTile, uTile, cTile)

            pto.tpush_to_aiv(cTile, SPLIT_NONE, id=ID_C2V)

            signal = pto.tpop_from_aiv(tile_type_signal_mat, SPLIT_NONE, id=ID_V2C)
            pto.tfree_from_aiv(SPLIT_NONE, id=ID_V2C)

    # ---------------------------------------------------------------
    # Vector kernel
    # ---------------------------------------------------------------
    @pto.func(kernel="vector")
    def vector_kernel(
        gm_slot_buffer: "ptr_type",
        y_ptr: "ptr_type",
        total_len_i32: "len_type",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cLAST_ROW_ELEM = const(TILE_SIZE - 1)
        cTILE_SIZE = const(TILE_SIZE)
        cN_TILE_ELEM = const(TILE_SIZE * TILE_SIZE)
        c0f = const(0.0, s.float32)
        c0i64 = const(0, s.int64)
        c0_i32 = const(0, type=i32)

        total_len = s.index_cast(total_len_i32)
        num_tiles = total_len // cN_TILE_ELEM

        # --- GM slot buffer partitioning (must match cube) ---
        gm_c2v = gm_slot_buffer
        gm_v2c = pto.add_ptr(gm_slot_buffer, const(GM_C2V_ELEMS))

        # --- Initialize C2V pipe (Vector is consumer, dir_mask=1) ---
        c2v_local = pto.reserve_buffer(
            name="c2v_fifo", size=C2V_FIFO_BYTES, location="VEC"
        )
        pto.aiv_initialize_pipe(
            id=ID_C2V,
            dir_mask=1,
            slot_size=SLOT_SIZE_C2V,
            gm_slot_buffer=gm_c2v,
            c2v_consumer_buf=c2v_local,
            v2c_consumer_buf=c0_i32,
        )

        # --- Initialize V2C pipe (Vector is producer, dir_mask=2) ---
        v2c_import = pto.import_reserved_buffer(
            name="v2c_fifo", peer_func="@cube_kernel"
        )
        pto.aiv_initialize_pipe(
            id=ID_V2C,
            dir_mask=2,
            slot_size=SLOT_SIZE_V2C,
            gm_slot_buffer=gm_v2c,
            c2v_consumer_buf=c0_i32,
            v2c_consumer_buf=v2c_import,
        )

        tvOut_vec = pto.as_tensor(
            tensor_type,
            ptr=y_ptr,
            shape=[total_len // cTILE_SIZE, cTILE_SIZE],
            strides=[cTILE_SIZE, c1],
        )

        rowTile = pto.alloc_tile(tile_type_row)
        sumTile1x8 = pto.alloc_tile(tile_type_1x8)
        signalTile = pto.alloc_tile(tile_type_1x8)
        tile.setval(sumTile1x8, c0, c0f)

        for tile_idx in pto.range(c0, num_tiles, c1):
            # Pop the ACC tile from Cube via C2V pipe.
            # The tile arrives in VEC space as a full TILE_SIZE x TILE_SIZE tile.
            cTile_vec = pto.tpop_from_aic(tile_type_c_vec, SPLIT_NONE, id=ID_C2V)

            vid = pto.get_subblock_idx()

            with pto.if_context(vid == c0i64):
                tile_offset = tile_idx * cTILE_SIZE

                # Step 1: Store the full popped tile to GM output.
                # This writes the within-tile prefix sums (matmul result).
                svOut_full = pto.slice_view(
                    subtensor_type_c,
                    source=tvOut_vec,
                    offsets=[tile_offset, c0],
                    sizes=[cTILE_SIZE, cTILE_SIZE],
                )
                pto.store(cTile_vec, svOut_full)

                # Step 2: Re-read rows from GM and add the running sum.
                for r in pto.range(c0, cTILE_SIZE, c1):
                    offset = tile_offset + r
                    svRow = pto.slice_view(
                        subtensor_type_row,
                        source=tvOut_vec,
                        offsets=[offset, c0],
                        sizes=[c1, cTILE_SIZE],
                    )

                    pto.load(svRow, rowTile)

                    # Extract the running sum from the persistent buffer
                    running_sum = tile.getval(sumTile1x8, c0, dtype=s.float32)

                    tile.adds(rowTile, running_sum, rowTile)
                    # Sync scalar pipe and vector pipe before extracting value
                    pto.barrier(pto.PIPE_ALL)
                    running_sum_next = tile.getval(
                        rowTile, cLAST_ROW_ELEM, dtype=s.float32
                    )
                    # Persist new running sum
                    tile.setval(sumTile1x8, c0, running_sum_next)

                    pto.store(rowTile, svRow)

            # Free the C2V slot after processing
            pto.tfree_from_aic(SPLIT_NONE, id=ID_C2V)

            # Signal Cube that this tile is done (V2C push)
            pto.tpush_to_aic(signalTile, SPLIT_NONE, id=ID_V2C)

    # ---------------------------------------------------------------
    # Entry point (dispatches to both kernels)
    # ---------------------------------------------------------------
    @pto.func
    def run_scan_tpushpop(
        ffts_addr: "ffts_type",
        gm_slot_buffer: "ptr_type",
        x_ptr: "ptr_type",
        y_ptr: "ptr_type",
        u_ptr: "ptr_type",
        total_len_i32: "len_type",
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(cube_kernel, gm_slot_buffer, x_ptr, u_ptr, total_len_i32)
        pto.call(vector_kernel, gm_slot_buffer, y_ptr, total_len_i32)


def test_scan(n_tiles=64):
    device = get_test_device()
    torch.npu.set_device(device)

    total_len = TILE_SIZE * TILE_SIZE * n_tiles
    torch.manual_seed(0)
    dtype = torch.float32

    # Prepare Inputs
    x = torch.rand(size=(total_len,), device=device, dtype=dtype).contiguous()
    y = torch.zeros_like(x)

    # Generate upper triangular matrix of 1s (s x s)
    u = torch.triu(
        torch.ones((TILE_SIZE, TILE_SIZE), device=device, dtype=dtype)
    ).contiguous()

    # GM slot buffer for TPush/TPop FIFO staging
    gm_slot_buffer = torch.zeros((GM_TOTAL_ELEMS,), dtype=torch.float32, device=device)

    # Expected PyTorch computation
    expected_scan = torch.cumsum(x.cpu(), dim=0)

    # NPU kernel execution — scan_module is a JitWrapper that lazily
    # compiles on first call; ffts_addr is injected automatically.
    repeat_runs = 20
    print(
        f"Running TPush/TPop scan for {total_len} elements "
        f"({n_tiles} {TILE_SIZE}x{TILE_SIZE} tiles)"
    )
    actual_scan = []
    for _ in range(repeat_runs):
        y.zero_()
        scan_module(gm_slot_buffer, x, y, u, total_len)
        actual_scan.append(y.cpu().clone())

    torch.npu.synchronize()

    # Check for consistency across runs and correctness against the expected count
    repeat_results = []
    for i, scan in enumerate(actual_scan):
        are_close = torch.allclose(scan, expected_scan, rtol=1e-3, atol=1e-3)
        if not are_close:
            unequal_count = torch.sum(scan != expected_scan)
        else:
            unequal_count = 0
        repeat_results.append([are_close, unequal_count])

    has_mismatch = any(not eq for eq, _ in repeat_results)
    if has_mismatch:
        print("Expected:\n", expected_scan[-10:])
        for i, result in enumerate(repeat_results):
            eq, count = result
            if not eq:
                print(
                    f"Inconsistent results run {i}, different elements: {count}/{total_len}. Sample:"
                )
                print(actual_scan[i][-10:])
        raise AssertionError(
            f"Scan mismatch for tile_size={TILE_SIZE}, total_len={total_len} ({n_tiles} tiles)"
        )

    print("All results matched. TPush/TPop scan test passed successfully.\n")


if __name__ == "__main__":
    test_scan(1)
    test_scan(16)
    test_scan(64)
    test_scan(100)
    test_scan(1000)
