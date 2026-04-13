// Bidirectional pipe example.
//
// This reduced version only uses the C2V pipe:
// - `c2v_fifo`: cube/kernel `@cube_kernel` pushes to vector/kernel `@vector_kernel`
//
// `gm_slot_buffer` is the GM-backed slot storage for these pipes. The reserve/import
// ops connect each side of the same named FIFO, and `aic/aiv_initialize_pipe`
// binds those FIFO endpoints to the shared GM slot buffer plus each side's local
// consumer buffer.
//
// End-to-end data flow:
// - Cube loads one input matrix `X` from GM.
// - Cube computes `Y = X @ X`.
// - Cube sends that accumulator tile to vector over `c2v_fifo`.
// - Vector pops the tile and stores it to GM as output matrix `Y`.
//
// What is transferred:
// - Cube -> Vector: one full `16 x 16` `f32` accumulator tile `Y = X @ X`
//   sent with `pto.tpush_to_aiv` using `split = 0` (no split). Vector receives
//   that same logical `16 x 16` tile with `pto.tpop_from_aic` in a vector tile
//   type/layout, then stores it to the GM output buffer.
//
// Shape summary:
// - All transferred tiles are `rows=16, cols=16, dtype=f32`
// - Cube-produced C2V tile: `loc=acc`, `blayout=col_major`, `slayout=row_major`
// - Vector-consumed tile after C2V pop: `loc=vec`, `blayout=row_major`, `slayout=none_box`
module {

  func.func @call_both(%gm_slot_buffer: !pto.ptr<f32>, %gm_x: !pto.ptr<f32>, %gm_y: !pto.ptr<f32>) attributes {pto.entry} {
    func.call @cube_kernel(%gm_slot_buffer, %gm_x) : (!pto.ptr<f32>, !pto.ptr<f32>) -> ()
    func.call @vector_kernel(%gm_slot_buffer, %gm_y) : (!pto.ptr<f32>, !pto.ptr<f32>) -> ()
    return
  }

  func.func @cube_kernel(%gm_slot_buffer: !pto.ptr<f32>, %gm_x: !pto.ptr<f32>) attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c2v_import = pto.import_reserved_buffer {
      name = "c2v_fifo",
      peer_func = @vector_kernel
    } -> i32
    %c0_i32 = arith.constant 0 : i32
    pto.aic_initialize_pipe {dir_mask = 1, slot_size = 1024}
      (gm_slot_buffer = %gm_slot_buffer : !pto.ptr<f32>,
       c2v_consumer_buf = %c2v_import : i32,
       v2c_consumer_buf = %c0_i32 : i32)

    %x_mat_tile = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %x_left_tile = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %x_right_tile = pto.alloc_tile : !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>
    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    %gm_x_view = pto.make_tensor_view %gm_x, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<?x?xf32>
    %gm_x_tile_view = pto.partition_view %gm_x_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tload ins(%gm_x_tile_view : !pto.partition_tensor_view<16x16xf32>) outs(%x_mat_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tmov ins(%x_mat_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%x_left_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tmov ins(%x_mat_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%x_right_tile : !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
    pto.tmatmul ins(%x_left_tile, %x_right_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>, !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>) outs(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
    pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) {split = 0}
    return
  }

  func.func @vector_kernel(%gm_slot_buffer: !pto.ptr<f32>, %gm_y: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c2v_local = pto.reserve_buffer {
      name = "c2v_fifo",
      size = 4096,
      location = #pto.address_space<vec>,
      auto = true
    } -> i32
    %c0_i32 = arith.constant 0 : i32
    pto.aiv_initialize_pipe {dir_mask = 1, slot_size = 1024}
      (gm_slot_buffer = %gm_slot_buffer : !pto.ptr<f32>,
       c2v_consumer_buf = %c2v_local : i32,
       v2c_consumer_buf = %c0_i32 : i32)

    %gm_y_view = pto.make_tensor_view %gm_y, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<?x?xf32>
    %gm_y_tile_view = pto.partition_view %gm_y_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x16xf32>
    %recv_tile = pto.tpop_from_aic {split = 0}
      -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tstore ins(%recv_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%gm_y_tile_view : !pto.partition_tensor_view<16x16xf32>)
    pto.tfree_from_aic {split = 0}
    return
  }

}
