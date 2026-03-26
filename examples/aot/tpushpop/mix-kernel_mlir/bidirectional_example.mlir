module {
  func.func @cube_kernel(%gm_slot_buffer: i32)
      attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %v2c_local = pto.reserve_buffer {
      name = "v2c_fifo",
      size = 4096,
      location = #pto.address_space<mat>,
      auto = true
    } -> i32
    %c2v_import = pto.import_reserved_buffer {
      name = "c2v_fifo",
      peer_func = @vector_kernel
    } -> i32
    pto.aic_initialize_pipe {dir_mask = 3, slot_size = 1024}
      (gm_slot_buffer = %gm_slot_buffer : i32,
       c2v_consumer_buf = %c2v_import : i32,
       v2c_consumer_buf = %v2c_local : i32)

    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) {split = 0}

    %mat_tile = pto.tpop_from_aiv {split = 1}
      -> !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>
    pto.tfree_from_aiv {split = 2}
    return
  }

  func.func @vector_kernel(%gm_slot_buffer: i32)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c2v_local = pto.reserve_buffer {
      name = "c2v_fifo",
      size = 4096,
      location = #pto.address_space<vec>,
      auto = true
    } -> i32
    %v2c_import = pto.import_reserved_buffer {
      name = "v2c_fifo",
      peer_func = @cube_kernel
    } -> i32
    pto.aiv_initialize_pipe {dir_mask = 3, slot_size = 1024}
      (gm_slot_buffer = %gm_slot_buffer : i32,
       c2v_consumer_buf = %c2v_local : i32,
       v2c_consumer_buf = %v2c_import : i32)

    %vec_tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tpush_to_aic(%vec_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {split = 0}

    %recv_tile = pto.tpop_from_aic {split = 1}
      -> !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tfree_from_aic {split = 2}
    return
  }

  func.func @cube_kernel_nested(%gm_slot_buffer: i32)
      attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %true = arith.constant true
    scf.if %true {
      %v2c_local = pto.reserve_buffer {
        name = "v2c_fifo_nested",
        size = 4096,
        location = #pto.address_space<mat>,
        auto = true
      } -> i32
      %c2v_import = pto.import_reserved_buffer {
        name = "c2v_fifo_nested",
        peer_func = @vector_kernel_nested
      } -> i32
      pto.aic_initialize_pipe {dir_mask = 3, slot_size = 1024}
        (gm_slot_buffer = %gm_slot_buffer : i32,
         c2v_consumer_buf = %c2v_import : i32,
         v2c_consumer_buf = %v2c_local : i32)

      %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
      pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) {split = 0}

      %recv_tile = pto.tpop_from_aiv {split = 1}
        -> !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>
      pto.tfree_from_aiv {split = 2}
    }
    return
  }

  func.func @vector_kernel_nested(%gm_slot_buffer: i32)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %true = arith.constant true
    scf.if %true {
      %c2v_local = pto.reserve_buffer {
        name = "c2v_fifo_nested",
        size = 4096,
        location = #pto.address_space<vec>,
        auto = true
      } -> i32
      %v2c_import = pto.import_reserved_buffer {
        name = "v2c_fifo_nested",
        peer_func = @cube_kernel_nested
      } -> i32
      pto.aiv_initialize_pipe {dir_mask = 3, slot_size = 1024}
        (gm_slot_buffer = %gm_slot_buffer : i32,
         c2v_consumer_buf = %c2v_local : i32,
         v2c_consumer_buf = %v2c_import : i32)

      %vec_tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tpush_to_aic(%vec_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {split = 0}

      %recv_tile = pto.tpop_from_aic {split = 1}
        -> !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tfree_from_aic {split = 2}
    }
    return
  }
}

// A3-LABEL: AICORE void cube_kernel(
// A3: auto {{v[0-9]+}} = TPipe<0, Direction::DIR_C2V, 1024, 4, 4>(
// A3: auto {{v[0-9]+}} = TPipe<2, Direction::DIR_V2C, 1024, 4, 4>(
// A3: TPUSH<TPipe<0, Direction::DIR_C2V, 1024, 4, 4>
// A3: TPOP<TPipe<2, Direction::DIR_V2C, 1024, 4, 4>
// A3: TFREE<TPipe<2, Direction::DIR_V2C, 1024, 4, 4>

// A3-LABEL: AICORE void vector_kernel(
// A3: auto {{v[0-9]+}} = TPipe<0, Direction::DIR_C2V, 1024, 4, 4>(
// A3: auto {{v[0-9]+}} = TPipe<2, Direction::DIR_V2C, 1024, 4, 4>(
// A3: TPUSH<TPipe<2, Direction::DIR_V2C, 1024, 4, 4>
// A3: TPOP<TPipe<0, Direction::DIR_C2V, 1024, 4, 4>
// A3: TFREE<TPipe<0, Direction::DIR_C2V, 1024, 4, 4>

// A3-LABEL: AICORE void cube_kernel_nested(
// A3: if (
// A3: auto {{v[0-9]+}} = TPipe<0, Direction::DIR_C2V, 1024, 4, 4>(
// A3: auto {{v[0-9]+}} = TPipe<2, Direction::DIR_V2C, 1024, 4, 4>(
// A3: TPUSH<TPipe<0, Direction::DIR_C2V, 1024, 4, 4>
// A3: TPOP<TPipe<2, Direction::DIR_V2C, 1024, 4, 4>
// A3: TFREE<TPipe<2, Direction::DIR_V2C, 1024, 4, 4>
