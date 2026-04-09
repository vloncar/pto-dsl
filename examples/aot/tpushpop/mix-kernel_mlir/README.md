# TPush / TPop mixed-kernel examples

Small examples of tile FIFO communication between Cube (`AIC`) and Vector (`AIV`).

```bash
python run.py c2v
python run.py v2c
python run.py bidi
```

`python run.py` defaults to `c2v`.

Files:

- `kernels/` has the Python builders.
- `build_artifacts/` gets generated MLIR, generated C++, and the `.so`.
- `gm_slot_buffer` is the GM backing store for the pipe.
- `caller.cpp` sets the FFTS base before launching the generated kernel.

Core idea:

- `aic_initialize_pipe` / `aiv_initialize_pipe` lower to matching `TPipe<...>` objects.
- `gm_slot_buffer` is the shared GM slot memory used by that `TPipe`.
- `tpush_to_aiv` / `tpush_to_aic` lower to `TPUSH(pipe, tile)`.
- `tpop_from_aic` / `tpop_from_aiv` lower to `TPOP(pipe, tile)`.
- `tfree_from_aic` / `tfree_from_aiv` lower to `TFREE(pipe)` and release the consumed slot.

## C2V

Cube sends. Vector receives.

This example computes `X @ X` on Cube, sends the accumulator tile to Vector, then Vector stores it to GM.

```text
Cube:   load X -> matmul -> tpush_to_aiv
Vector: tpop_from_aic -> store Y -> tfree_from_aic
```

Pipe wiring:

- Vector owns the consumer buffer: `reserve_buffer("c2v_fifo", location="VEC")`
- Cube imports it: `import_reserved_buffer("c2v_fifo", peer_func="@vector_kernel")`
- Both sides initialize with `dir_mask = 1`

## V2C

Vector sends. Cube receives.

This example loads `X` on Vector, sends that tile to Cube, then Cube stores it to GM.

```text
Vector: load X -> tpush_to_aic
Cube:   tpop_from_aiv -> store Y -> tfree_from_aiv
```

Pipe wiring:

- Cube owns the consumer buffer: `reserve_buffer("v2c_fifo", location="MAT")`
- Vector imports it: `import_reserved_buffer("v2c_fifo", peer_func="@cube_kernel")`
- Both sides initialize with `dir_mask = 2`

## BIDI

Both directions are enabled.

This example sends `X @ X` from Cube to Vector. Vector doubles it and sends it back. Cube receives the returned tile and stores it to GM.

```text
Cube:   matmul -> tpush_to_aiv
Vector: tpop_from_aic -> add -> tpush_to_aic -> tfree_from_aic
Cube:   tpop_from_aiv -> store Y -> tfree_from_aiv
```

Pipe wiring:

- Vector reserves `c2v_fifo`; Cube imports it
- Cube reserves `v2c_fifo`; Vector imports it
- Both sides initialize with `dir_mask = 3`

For `dir_mask = 3`, allocate FIFO backing for both directions. `run.py` uses `8 KiB`.
