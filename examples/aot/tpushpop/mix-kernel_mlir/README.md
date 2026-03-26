# Bidirectional `TPUSH`/`TPOP` MLIR Example

This example mirrors the `mix-kernel_cpp` flow, but starts from
[`bidirectional_example.mlir`](/home/fskogh/pto-dsl/examples/aot/tpushpop/mix-kernel_mlir/bidirectional_example.mlir).

The pipeline is:

1. run `ptoas --pto-arch=a3 bidirectional_example.mlir > build_artifacts/bidirectional_example.cpp`
2. compile the generated C++ together with `caller.cpp`
3. build `./tpushpop_mlir_lib.so`
4. launch the generated `pto.entry` kernel from Python

## Run

```bash
python run_bidirectional_example.py
```
