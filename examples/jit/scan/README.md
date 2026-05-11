# Single core prefix sum (scan)

An implementation of prefix sum (scan) algorithm, based on https://arxiv.org/abs/2505.15112v1. Only the single core algorithm is implemented (ScanU from the paper).

## Algorithm

The ScanU algorithm computes the prefix sum (cumulative sum) of a 1-D input vector by decomposing it into tiles and leveraging the Cube unit's matrix multiply for parallelism within each tile:

1. **Reshape** the flat input vector into tiles of shape `TILE_SIZE × TILE_SIZE`.
2. **Precompute** an upper-triangular matrix `U` of 1s (shape `TILE_SIZE × TILE_SIZE`).
3. **For each tile** `X_i`:
   - **Cube** computes `C_i = X_i @ U`. Each row of `C_i` contains partial prefix sums within that row of the tile.
   - **Vector** adds a running scalar sum to every element of each row (the cross-tile carry), then extracts the last element of the row as the new running sum.
4. The concatenation of all processed tiles is the full prefix sum.

The Cube and Vector units run concurrently but must synchronize: the Cube cannot overwrite the next tile's result before the Vector has finished processing the current one.

## Implementations

There are two implementations that differ only in how Cube ↔ Vector synchronization is achieved. The algorithm logic, tile types, memory layouts, and test harness are identical.

### `run_scan_single_core.py` — TSync (sync_set / sync_wait)

Uses a single function with `cube_section` / `vector_section` blocks. Synchronization is performed with the low-level `sync_set` / `sync_wait` primitives operating on `PIPE_FIX` and `PIPE_MTE3`:

- Cube signals Vector via `pto.sync_set(pto.PIPE_FIX, 0)` after storing the matmul result to GM.
- Vector waits via `pto.sync_wait(pto.PIPE_FIX, 0)`, processes the tile, then signals back via `pto.sync_set(pto.PIPE_MTE3, 1)`.
- Cube waits for the Vector's acknowledgement via `pto.sync_wait(pto.PIPE_MTE3, 1)` before advancing.

```bash
python ./run_scan_single_core.py
```

### `run_scan_single_core_tpushpop.py` — TPush / TPop

Uses the structured multi-function module pattern with separate `cube_kernel` and `vector_kernel` functions. Synchronization uses two unidirectional TPush/TPop pipes:

| Pipe | Direction | `dir_mask` | Purpose |
|------|-----------|------------|---------|
| C2V  | Cube → Vector | 1 | Sends the matmul ACC tile directly to Vector VEC memory |
| V2C  | Vector → Cube | 2 | Sends a dummy signal tile back for back-pressure |

Both pipes use GM-staged L2G2L transport with `slot_num=8`.

The Cube pushes the ACC tile via `tpush_to_aiv`, then blocks on `tpop_from_aiv` (waiting for the Vector's V2C signal). The Vector pops with `tpop_from_aic`, stores the tile to GM, processes rows with the running sum, then pushes a signal via `tpush_to_aic` and frees the C2V slot with `tfree_from_aic`.

```bash
python ./run_scan_single_core_tpushpop.py
```

## Differences between TSync and TPush/TPop

| Aspect | TSync | TPush/TPop |
|--------|-------|------------|
| Code structure | Single function with `cube_section` / `vector_section` | Separate `@pto.func(kernel="cube")` and `@pto.func(kernel="vector")` functions |
| Sync mechanism | `sync_set` / `sync_wait` on hardware pipes | `tpush` / `tpop` / `tfree` on logical pipe handles |
| Data transfer | Cube stores to GM, Vector loads from GM | Cube pushes ACC tile through pipe, Vector pops into VEC memory, then stores to GM |
| Back-pressure | Explicit `sync_wait` on `PIPE_MTE3` | V2C pipe with dummy signal tile |
| GM slot buffer | Not needed | Required (FIFO staging area for both pipes) |
| Address management | N/A | `reserve_buffer` / `import_reserved_buffer` for cross-kernel FIFO address sharing |
| Insert sync | `enable_insert_sync=False` (manual sync) | `--enable-insert-sync` (PTOAS auto-inserts intra-pipe sync) |

## Implementation notes

- The running sum calculation is stored in a tile (`sumTile1x8`) to avoid issues with the compiler removing it as unused code.
- Due to the inability to synchronize on the scalar pipe with `record_wait_pair`, a `barrier(PIPE_ALL)` is used as a workaround before extracting the last row element.
- **`record_wait_pair` and `--enable-insert-sync` are mutually exclusive.** The TSync version uses manual sync (`enable_insert_sync=False` + explicit `record_wait_pair` calls). The TPush/TPop version uses auto sync (`--enable-insert-sync`; PTOAS InsertSync pass inserts all needed `set_flag`/`wait_flag` pairs). Mixing both causes event ID collisions that lead to data races and non-deterministic results.
