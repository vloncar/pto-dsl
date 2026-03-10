from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

# 32 KB of UB
_TILE_SIZE_BYTES = 32 * 1024
_DTYPE_BYTES = {"fp16": 2, "fp32": 4}


def meta_data_row(dtype="fp32"):
    pto_dtype = {"fp16": pto.float16, "fp32": pto.float32}[dtype]
    elements_per_tile = _TILE_SIZE_BYTES // _DTYPE_BYTES[dtype]
    ptr_type = pto.PtrType(pto_dtype)
    index_dtype = pto.int32

    tensor_type = pto.TensorType(rank=1, dtype=pto_dtype)
    subtensor_in = pto.SubTensorType(shape=[1, elements_per_tile], dtype=pto_dtype)

    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[1, elements_per_tile],
        valid_shape=[1, -1],
        dtype=pto_dtype,
        memory_space="VEC",
        config=tile_cfg,
    )

    return {
        "ptr_type": ptr_type,
        "pto_dtype": pto_dtype,
        "elements_per_tile": elements_per_tile,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "subtensor_in": subtensor_in,
        "tile_type": tile_type,
    }


def build_rowsum(dtype="fp32"):
    """
    Computes per-row sum across columns using PTO TROWSUM (`tile.row_sum` wrapper).

    Args:
        x_ptr : dtype[batch * n_cols]    input matrix flattened row-major
        y_ptr : dtype[batch]             output vector, one sum per row
        batch : int32
        n_cols: int32 (<= elements_per_tile)

    Semantics:
        y[row] = sum_{j=0..n_cols-1} x[row, j]
    """
    _meta_data = lambda: meta_data_row(dtype=dtype)

    @to_ir_module(meta_data=_meta_data)
    def _kernel(
        x_ptr: "ptr_type",
        y_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            num_cores = s.index_cast(pto.get_block_num())

            rows_per_core = s.ceil_div(batch, num_cores)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, batch)
            num_rows = row_end - row_start

            total_elems = batch * n_cols
            tv_x = pto.as_tensor(
                tensor_type, ptr=x_ptr, shape=[total_elems], strides=[c1]
            )
            tv_y = pto.as_tensor(
                tensor_type, ptr=y_ptr, shape=[batch], strides=[c1]
            )

            with pto.if_context(num_rows > c0):
                tb_x = pto.alloc_tile(tile_type, valid_col=n_cols)
                tb_sum = pto.alloc_tile(
                    tile_type, valid_col=c1
                )  # scalar output
                tb_tmp = pto.alloc_tile(
                    tile_type, valid_col=n_cols
                )  # scratch

                for r in pto.range(c0, num_rows, c1):
                    gm_offset = (row_start + r) * n_cols

                    sv_x = pto.slice_view(
                        subtensor_in,
                        source=tv_x,
                        offsets=[gm_offset],
                        sizes=[n_cols],
                    )

                    # y is a vector of length batch; write one element per row
                    sv_y = pto.slice_view(
                        subtensor_in,
                        source=tv_y,
                        offsets=[row_start + r],
                        sizes=[c1],
                    )

                    pto.load(sv_x, tb_x)
                    tile.row_sum(tb_x, tb_tmp, tb_sum)

                    # Store the 1-element tile to y[row]
                    pto.store(tb_sum, sv_y)

    return _kernel


def meta_data_col(dtype="fp32"):
    pto_dtype = {"fp16": pto.float16, "fp32": pto.float32}[dtype]
    ptr_type = pto.PtrType(pto_dtype)
    index_dtype = pto.int32

    tile_rows = 32
    tile_cols = 32

    tensor2d_type = pto.TensorType(rank=2, dtype=pto_dtype)
    subtensor_in = pto.SubTensorType(shape=[tile_rows, tile_cols], dtype=pto_dtype)
    subtensor_out = pto.SubTensorType(shape=[1, tile_cols], dtype=pto_dtype)

    tile_cfg = pto.TileBufConfig()

    # Dynamic valid rows + cols
    tile_type = pto.TileBufType(
        shape=[tile_rows, tile_cols],
        valid_shape=[-1, -1],
        dtype=pto_dtype,
        memory_space="VEC",
        config=tile_cfg,
    )

    # Dynamic valid cols
    tile_sum_type = pto.TileBufType(
        shape=[1, tile_cols],
        valid_shape=[1, -1],
        dtype=pto_dtype,
        memory_space="VEC",
        config=tile_cfg,
    )

    return {
        "ptr_type": ptr_type,
        "pto_dtype": pto_dtype,
        "index_dtype": index_dtype,
        "tensor2d_type": tensor2d_type,
        "subtensor_in": subtensor_in,
        "subtensor_out": subtensor_out,
        "tile_type": tile_type,
        "tile_sum_type": tile_sum_type,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
    }


def build_colsum(dtype="fp32"):
    _meta_data = lambda: meta_data_col(dtype=dtype)

    @to_ir_module(meta_data=_meta_data)
    def _kernel(
        x_ptr: "ptr_type",
        y_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile_rows = const(tile_rows)
        c_tile_cols = const(tile_cols)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            num_cores = s.index_cast(pto.get_block_num())

            cols_per_core = s.ceil_div(n_cols, num_cores)
            col_start = bid * cols_per_core
            col_end = s.min_u(col_start + cols_per_core, n_cols)
            num_cols = col_end - col_start

            tv_x = pto.as_tensor(
                tensor2d_type,
                ptr=x_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )
            tv_y = pto.as_tensor(
                tensor2d_type,
                ptr=y_ptr,
                shape=[c1, n_cols],
                strides=[n_cols, c1],
            )

            with pto.if_context(num_cols > c0):
                for col in pto.range(col_start, col_end, c_tile_cols):
                    cols_this = s.min_u(c_tile_cols, col_end - col)

                    with pto.if_context(batch > c0):
                        first_rows = s.min_u(c_tile_rows, batch)

                        tb_x = pto.alloc_tile(
                            tile_type,
                            valid_row=first_rows,
                            valid_col=cols_this,
                        )
                        tb_tmp = pto.alloc_tile(
                            tile_type,
                            valid_row=first_rows,
                            valid_col=cols_this,
                        )
                        tb_acc = pto.alloc_tile(
                            tile_sum_type,
                            valid_col=cols_this,
                        )

                        sv_x0 = pto.slice_view(
                            subtensor_in,
                            source=tv_x,
                            offsets=[c0, col],
                            sizes=[first_rows, cols_this],
                        )

                        pto.load(sv_x0, tb_x)
                        tile.col_sum(tb_x, tb_tmp, tb_acc)

                        for row in pto.range(c_tile_rows, batch, c_tile_rows):
                            rows_this = s.min_u(c_tile_rows, batch - row)

                            tb_x_k = pto.alloc_tile(
                                tile_type,
                                valid_row=rows_this,
                                valid_col=cols_this,
                            )
                            tb_tmp_k = pto.alloc_tile(
                                tile_type,
                                valid_row=rows_this,
                                valid_col=cols_this,
                            )
                            tb_part = pto.alloc_tile(
                                tile_sum_type,
                                valid_col=cols_this,
                            )

                            sv_xk = pto.slice_view(
                                subtensor_in,
                                source=tv_x,
                                offsets=[row, col],
                                sizes=[rows_this, cols_this],
                            )

                            pto.load(sv_xk, tb_x_k)
                            tile.col_sum(tb_x_k, tb_tmp_k, tb_part)
                            tile.add(tb_acc, tb_part, tb_acc)

                        sv_y = pto.slice_view(
                            subtensor_out,
                            source=tv_y,
                            offsets=[c0, col],
                            sizes=[c1, cols_this],
                        )
                        pto.store(tb_acc, sv_y)

    return _kernel
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["row", "col"], default="row")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()
    builder = build_rowsum if args.mode == "row" else build_colsum
    print(builder(dtype=args.dtype))
