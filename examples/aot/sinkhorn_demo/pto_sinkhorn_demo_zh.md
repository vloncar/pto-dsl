# 使用PTO指令集十分钟快速开发DeepSeek-V4 mHC中的Sinkhorn-Knopp算子

DeepSeek-V4使用了[mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)，其中的[Sinkhorn-Knopp迭代](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem)需要专门的融合算子。在刚刚开源的[TileKernels](https://github.com/deepseek-ai/TileKernels)仓中包含了TileLang(GPU)的实现 [sinkhorn_kernel.py](https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/sinkhorn_kernel.py)，核心的`mhc_sinkhorn_kernel`大约20行Python。我们基于PTO指令集的Python接口，也用20行左右的Python表达其核心计算逻辑，用几分钟时间快速完成昇腾上的Sinkhorn算子开发，并验证通过TileKernels仓里的单元测试[test_sinkhorn.py](https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tests/mhc/test_sinkhorn.py)。

- 完整代码见 [sinkhorn_demo](https://github.com/huawei-csl/pto-dsl/tree/0.1.2/examples/aot/sinkhorn_demo)
- 核心算子部分见 [sinkhorn_k4_builder.py](https://github.com/huawei-csl/pto-dsl/blob/0.1.2/examples/aot/sinkhorn_demo/sinkhorn_k4_builder.py#L127-L147)

Sinkhorn迭代的数学公式很简单，参考单元测试的torch基线代码[sinkhorn_normalize_ref](https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/torch/mhc.py#L8):

```python
def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x
```

x为3D张量，shape `[batch, 4, 4]`. `4`是mHC论文中选用的扩展系数，可作为编译时的常量；`batch`和输入token数成正比，需要以动态维度处理。算法分两个阶段：
- 先对x按行做softmax
- 然后反复做行和列的归一化，使得各行各列的和收敛到1 (有这个性质的矩阵称为[Doubly stochastic matrix](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix))

先用`tload`把一块数据从HBM读取到buffer，然后在buffer上完成两个阶段的运算阶段。

第一阶段`x.softmax(-1)` 需要 [Numerically Stable Softmax](https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html#numerical-stability-of-softmax)，先除去每行的最大值(`row_max`, 对数空间的`row_expand_sub`)，然后再算softmax (`exp`, `row_sum`, `row_expand_div`)。最后的 `adds` 对应 `+ eps`。

```python
tile.row_max(mat_kk, scratch_kk, row_stat)
tile.row_expand_sub(mat_kk, row_stat, mat_kk)
tile.exp(mat_kk, mat_kk)

tile.row_sum(mat_kk, scratch_kk, row_stat)
tile.row_expand_div(mat_kk, row_stat, mat_kk)
tile.adds(mat_eps_rows, eps_h, mat_eps_rows)
```

(变量名`mat_kk`表示shape`[k, k]`的局部tile，`scratch_kk`用于存储shape`[k, k]`的临时结果)

第二阶段的行列归一化，也很直观，顺序调用`col_sum`, `col_expand_div`, `row_sum`, `row_expand_div`即可：

```python
tile.adds(mat_eps_rows, eps_h, mat_eps_rows)

tile.col_sum(mat_kk, scratch_kk, col_stat)
tile.adds(col_stat, eps_h, col_stat)
tile.col_expand_div(mat_kk, col_stat, mat_kk)

for _ in pto.range(c1, repeat_idx, c1):
    tile.row_sum(mat_kk, scratch_kk, row_stat)
    tile.adds(row_stat, eps_h, row_stat)
    tile.row_expand_div(mat_kk, row_stat, mat_kk)

    tile.col_sum(mat_kk, scratch_kk, col_stat)
    tile.adds(col_stat, eps_h, col_stat)
    tile.col_expand_div(mat_kk, col_stat, mat_kk)
```

这样我们就快速完成了Sinkhorn算子开发。 [test_sinkhorn.py](https://github.com/huawei-csl/pto-dsl/blob/0.1.2/examples/aot/sinkhorn_demo/test_sinkhorn.py) 脚本对接torch-npu输入，通过精度测试。
