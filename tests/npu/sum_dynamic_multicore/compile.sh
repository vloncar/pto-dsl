#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TMP=$(mktemp -d)
trap "rm -rf \"$TMP\"" EXIT

BISHENG_FLAGS=(
    -I${ASCEND_TOOLKIT_HOME}/include
    -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong
    -xcce -Xhost-start -Xhost-end
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    --npu-arch=dav-2201 -DMEMORY_BASE
    -std=gnu++17
)

for MODE in row col; do
    python "$SCRIPT_DIR/gen_ir.py" --mode "$MODE" > "$TMP/${MODE}sum.pto"
    ptoas --enable-insert-sync "$TMP/${MODE}sum.pto" -o "$TMP/${MODE}sum.cpp"

    python "$SCRIPT_DIR/caller.py" --mode "$MODE" > "$TMP/${MODE}sum_caller.cpp"

    bisheng "${BISHENG_FLAGS[@]}" \
        "$TMP/${MODE}sum_caller.cpp" \
        -o "$SCRIPT_DIR/${MODE}sum_lib.so"

    echo "Built ${MODE}sum_lib.so successfully."
done
