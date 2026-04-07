from mlir.dialects import arith, func, pto as _pto
from mlir.ir import (
    Attribute,
    Context,
    FlatSymbolRefAttr,
    InsertionPoint,
    Location,
    Module,
    Operation,
    UnitAttr,
)

from ptodsl import pto, tile
from ptodsl import scalar as s

const = s.const


def _call(name, *args):
    return Operation.create(
        "func.call",
        operands=list(args),
        attributes={"callee": FlatSymbolRefAttr.get(name)},
    )


def _kernel(fn, kind):
    fn.operation.attributes["pto.kernel_kind"] = Attribute.parse(
        f"#pto.kernel_kind<{kind}>"
    )


def build_module():
    with Context() as ctx, Location.unknown():
        _pto.register_dialect(ctx, load=True)
        module = Module.create()

        dtype = pto.float32
        ptr_ty = pto.PtrType(dtype)
        i32 = pto.int32
        tensor_ty = pto.TensorType(rank=2, dtype=dtype)
        tile_view_ty = pto.SubTensorType(shape=[16, 16], dtype=dtype)
        left_cfg = pto.TileBufConfig(blayout="ColMajor", slayout="RowMajor")
        x_mat_ty = pto.TileBufType(shape=[16, 16], dtype=dtype, memory_space="MAT")
        x_left_ty = pto.TileBufType(
            shape=[16, 16],
            dtype=dtype,
            memory_space="LEFT",
            config=left_cfg,
        )
        x_right_ty = pto.TileBufType(shape=[16, 16], dtype=dtype, memory_space="RIGHT")
        acc_ty = pto.TileBufType(shape=[16, 16], dtype=dtype, memory_space="ACC")
        recv_ty = pto.TileBufType(shape=[16, 16], dtype=dtype, memory_space="VEC")
        call_both_ty = func.FunctionType.get([ptr_ty, ptr_ty, ptr_ty], [])
        two_ptr_ty = func.FunctionType.get([ptr_ty, ptr_ty], [])

        with InsertionPoint(module.body):
            call_both = func.FuncOp("call_both", call_both_ty)
            cube_kernel = func.FuncOp("cube_kernel", two_ptr_ty)
            vector_kernel = func.FuncOp("vector_kernel", two_ptr_ty)

        call_both.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
        _kernel(cube_kernel, "cube")
        _kernel(vector_kernel, "vector")

        call_both_entry = call_both.add_entry_block()
        with InsertionPoint(call_both_entry):
            gm_slot_buffer, gm_x, gm_y = call_both_entry.arguments
            _call("cube_kernel", gm_slot_buffer, gm_x)
            _call("vector_kernel", gm_slot_buffer, gm_y)
            func.ReturnOp([])

        cube_entry = cube_kernel.add_entry_block()
        with InsertionPoint(cube_entry):
            gm_slot_buffer, gm_x = cube_entry.arguments
            c0 = const(0)
            c1 = const(1)
            c16 = const(16)
            c0_i32 = arith.ConstantOp(i32, 0).result
            c2v_import = pto.import_reserved_buffer(
                name="c2v_fifo",
                peer_func="@vector_kernel",
            )

            pto.aic_initialize_pipe(
                dir_mask=1,
                slot_size=1024,
                gm_slot_buffer=gm_slot_buffer,
                c2v_consumer_buf=c2v_import,
                v2c_consumer_buf=c0_i32,
            )

            x_mat_tile = pto.alloc_tile(x_mat_ty)
            x_left_tile = pto.alloc_tile(x_left_ty)
            x_right_tile = pto.alloc_tile(x_right_ty)
            acc_tile = pto.alloc_tile(acc_ty)

            gm_x_view = pto.as_tensor(
                tensor_ty,
                ptr=gm_x,
                shape=[c16, c16],
                strides=[c16, c1],
            )
            gm_x_tile_view = pto.slice_view(
                tile_view_ty,
                source=gm_x_view,
                offsets=[c0, c0],
                sizes=[c16, c16],
            )

            pto.load(gm_x_tile_view, x_mat_tile)
            tile.mov(x_mat_tile, x_left_tile)
            tile.mov(x_mat_tile, x_right_tile)
            tile.matmul(x_left_tile, x_right_tile, acc_tile)
            pto.tpush_to_aiv(acc_tile, 0)
            func.ReturnOp([])

        vector_entry = vector_kernel.add_entry_block()
        with InsertionPoint(vector_entry):
            gm_slot_buffer, gm_y = vector_entry.arguments
            c0 = const(0)
            c1 = const(1)
            c16 = const(16)
            c0_i32 = arith.ConstantOp(i32, 0).result
            c2v_local = pto.reserve_buffer(
                name="c2v_fifo",
                size=4096,
                location="VEC",
            )

            pto.aiv_initialize_pipe(
                dir_mask=1,
                slot_size=1024,
                gm_slot_buffer=gm_slot_buffer,
                c2v_consumer_buf=c2v_local,
                v2c_consumer_buf=c0_i32,
            )

            gm_y_view = pto.as_tensor(
                tensor_ty,
                ptr=gm_y,
                shape=[c16, c16],
                strides=[c16, c1],
            )
            gm_y_tile_view = pto.slice_view(
                tile_view_ty,
                source=gm_y_view,
                offsets=[c0, c0],
                sizes=[c16, c16],
            )

            recv_tile = pto.tpop_from_aic(recv_ty, 0)
            pto.store(recv_tile, gm_y_tile_view)
            pto.tfree_from_aic(0)
            func.ReturnOp([])

        module.operation.verify()
        return module


module = build_module()


if __name__ == "__main__":
    print(module)
