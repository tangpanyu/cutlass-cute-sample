# `gemm_opt_final` 中 CUTE 函数说明

本文档按 `gemm-opt.cu` 里的 `gemm_opt_final` 执行顺序，解释其中用到的 CUTE API 在做什么，以及这些 API 生成的 tensor shape 如何理解。

默认结合当前配置理解：

```cpp
config::GemmConfig<T, 128, 128, 32, 3, 4, 4>
```

也就是：

```text
kTileM = 128
kTileN = 128
kTileK = 32
kStage = 3
```

MMA atom 是：

```cpp
SM80_16x8x16_F16F16F16F16_TN
```

`make_tiled_mma` 后，一个 tiled MMA 的逻辑计算 tile 是：

```text
TiledShape_MNK = (32, 32, 16)
```

因此一个 CTA 的 `128 x 128 x 32` tile 在逻辑 tiled-MMA tile 粒度上会拆成：

```text
MMA_M = 128 / 32 = 4
MMA_N = 128 / 32 = 4
MMA_K =  32 / 16 = 2
```

注意：CUTE 打印出来的 shape 不是单纯的矩阵形状，而是“当前线程视角下”的 fragment/copy 视图。`MMA` 和 `CPY` 维度通常表示每个线程持有或每次 copy atom 操作的数据形状。比如 `tCrB` 打印出的 `_8` 是 B fragment 在当前线程视图里的 N 向重复维，不能直接和逻辑 `128 / 32 = 4` 画等号。

## 1. shared memory 大小相关

### `cute::cosize(layout)`

代码：

```cpp
T *Bshm = shm_data + cute::cosize(SmemLayoutA{});
```

`cosize` 返回一个 layout 覆盖的最大线性地址范围，也就是这个 layout 至少需要多少个元素的存储空间。

它和 `size(layout)` 不完全一样：

```text
size(layout)   = 逻辑元素个数
cosize(layout) = layout 映射到线性存储后需要覆盖的容量
```

对于普通连续 layout 二者通常一样；对于带 swizzle、stride、padding 的 layout，`cosize` 更适合用来分配 shared memory。

这里 `Bshm` 紧跟在 `A` 的 shared memory 区域之后：

```text
Ashm = shm_data
Bshm = shm_data + cosize(SmemLayoutA)
```

## 2. 构造 global memory Tensor

### `make_gmem_ptr(ptr)`

代码：

```cpp
make_gmem_ptr((T *)Aptr)
```

把普通裸指针包装成 CUTE 的 global memory pointer。这样 CUTE 后续的 `copy`、`tensor`、layout 逻辑知道这个数据来自 global memory。

它不移动数据，只是给指针加类型语义。

### `make_shape(...)`

代码：

```cpp
make_shape(m, k)
make_shape(n, k)
make_shape(m, n)
```

创建 tensor 的逻辑形状。例如：

```cpp
make_shape(m, k)
```

表示二维矩阵形状 `(m, k)`。

CUTE 的 shape 可以同时包含运行时整数和编译期整数：

```cpp
make_shape(m, k)                  // runtime shape
make_shape(Int<128>{}, Int<32>{}) // compile-time shape
```

### `make_stride(...)`

代码：

```cpp
make_stride(k, Int<1>{})
make_stride(n, Int<1>{})
```

定义逻辑坐标到线性地址的映射步长。

例如 A 是 row-major 的 `(M, K)`：

```cpp
A(i, j) -> base + i * k + j
```

所以 stride 是：

```text
(k, 1)
```

`Int<1>{}` 是 CUTE 的编译期整数，和普通 `1` 的含义一样，但它能参与更多编译期 layout 推导。

### `make_tensor(ptr, shape, stride)`

代码：

```cpp
Tensor A = make_tensor(make_gmem_ptr((T *)Aptr),
                       make_shape(m, k),
                       make_stride(k, Int<1>{}));
```

构造一个 CUTE tensor。Tensor 本质上是：

```text
pointer + layout
```

其中 layout 由 shape 和 stride 决定。这个对象不拥有数据，只是一个 view。

此处得到：

```text
A : (M, K)
B : (N, K)
D : (M, N)
```

注意这里 B 被建成 `(N, K)`，因为 kernel 使用的是 TN 形式的 MMA，B 按 `(n, k)` 访问。

## 3. 切出当前 CTA 负责的 global tile

### `make_tile(...)`

代码：

```cpp
make_tile(Int<kTileM>{}, Int<kTileK>{})
make_tile(Int<kTileN>{}, Int<kTileK>{})
make_tile(Int<kTileM>{}, Int<kTileN>{})
```

描述 tile 的大小。例如：

```text
A tile = (128, 32)
B tile = (128, 32)
D tile = (128, 128)
```

`make_tile` 和 `make_shape` 很像，但语义上更偏向“我要按这个大小切块”。

### `make_coord(...)`

代码：

```cpp
make_coord(iy, _)
make_coord(ix, _)
make_coord(iy, ix)
```

描述当前要取第几个 tile。

`_` 是 CUTE 的占位符，表示这个维度保留为后续遍历维度，不固定成某一个坐标。

例如：

```cpp
make_coord(iy, _)
```

表示：

```text
M 方向固定取第 iy 个 128 行 tile
K 方向不固定，保留下来作为 tile 序列维度
```

### `local_tile(tensor, tile_shape, tile_coord)`

代码：

```cpp
Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                       make_coord(iy, _));
```

从大 tensor 中切出当前 CTA 要处理的 tile view，类似于pytorch的View。

结果：

```text
gA : (kTileM, kTileK, k_tile_count)
gB : (kTileN, kTileK, k_tile_count)
gD : (kTileM, kTileN)
```

以 A 为例：

```text
A  : (M, K)
gA : (128, 32, K / 32)
```

前两个维度是一个 tile 内的局部坐标，第三维是 K 方向有多少个 tile。

## 4. 构造 shared memory Tensor

### `make_smem_ptr(ptr)`

代码：

```cpp
make_smem_ptr(Ashm)
make_smem_ptr(Bshm)
```

把裸指针包装成 CUTE 的 shared memory pointer。后续 CUTE copy 会据此选择 shared memory 访问路径。

它也不移动数据，只是提供地址空间类型信息。

### `make_tensor(make_smem_ptr(...), SmemLayout{})`

代码：

```cpp
auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});
```

用 shared memory pointer 加 shared memory layout 构造 tensor view。

当前 layout 是三维：

```text
sA : (kTileM, kTileK, kStage)
sB : (kTileN, kTileK, kStage)
```

第三维 `kStage` 是 pipeline 环形缓冲区。global 到 shared 的 `cp.async` 会提前写入后面的 stage，MMA 则从当前 `ismem_read` stage 读取。

`SmemLayoutA/B` 内部带有 `Swizzle`，用于减少 shared memory bank conflict。

## 5. MMA 线程切片和寄存器 fragment

### `TiledMMA tiled_mma`

代码：

```cpp
TiledMMA tiled_mma;
```

`TiledMMA` 是由 `make_tiled_mma` 生成的类型，描述整个 CTA 内线程如何协作执行一组 MMA atom。

它包含：

```text
MMA atom 的形状
线程到 MMA 子块的映射
每个线程持有哪些 fragment value
```

### `tiled_mma.get_slice(idx)`

代码：

```cpp
auto thr_mma = tiled_mma.get_slice(idx);
```

取当前线程 `threadIdx.x` 对应的 MMA 视图。

`thr_mma` 后续可以把 A/B/C tensor 分区成当前线程负责的 fragment。

### `thr_mma.partition_fragment_A(tensor)`

代码：

```cpp
auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
```

把 A 的一个 `(kTileM, kTileK)` global tile，按照 tiled MMA 的 A operand 布局，划分成当前线程的 register fragment。

返回的是 register-backed fragment tensor，不直接引用 global memory。

典型 shape：

```text
tCrA : (MMA, MMA_M, MMA_K)
     : ((_2,_2,_2), _4, _2)
```

解释：

```text
MMA   = 当前线程在一个 A tiled-MMA 子块中持有的值
MMA_M = CTA 的 M 方向有几个 tiled-MMA 子块，128 / 32 = 4
MMA_K = CTA 的 K 方向有几个 tiled-MMA 子块，32 / 16 = 2
```

`((_2,_2,_2))` 的 product 是 8，表示每线程在这个 A fragment 视图下有 8 个基本值。

### `thr_mma.partition_fragment_B(tensor)`

代码：

```cpp
auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
```

和 A 类似，把 B 的一个 `(kTileN, kTileK)` tile 分成当前线程的 B register fragment。

典型 shape：

```text
tCrB : (MMA, MMA_N, MMA_K)
     : ((_2,_2), _8, _2)
```

`MMA_N` 是 N 方向 tiled-MMA 子块重复次数。由于当前 tiled MMA 的 N 方向展开方式来自 atom layout 和 value layout，打印出来可能是 `_8`，不一定只看 `128 / 32`。

### `thr_mma.partition_fragment_C(tensor)`

代码：

```cpp
auto tCrD = thr_mma.partition_fragment_C(gD);
```

为当前线程创建 C/D accumulator fragment。

它表示当前线程负责累加的 C 元素，存放在 register 中。

典型 shape：

```text
tCrD : (MMA, MMA_M, MMA_N)
     : ((_2,_2), _4, _8)
```

第一个 `MMA` 维表示当前线程在一个 C MMA fragment 内持有的 accumulator value。

### `clear(tensor)`

代码：

```cpp
clear(tCrD);
```

把 tensor 中的所有元素清零。这里是把 accumulator fragment 置零：

```text
tCrD = 0
```

后续每次 `cute::gemm` 都会累加到 `tCrD`。

## 6. G2S：global memory 到 shared memory

### `G2SCopyA/B`

代码：

```cpp
G2SCopyA g2s_tiled_copy_a;
G2SCopyB g2s_tiled_copy_b;
```

这两个类型来自 config：

```cpp
make_tiled_copy(
  Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, T>{},
  make_layout(Shape<_32,_4>, Stride<_4,_1>),
  make_layout(Shape<_1,_8>)
)
```

它描述一个 CTA 内线程如何用 `cp.async` 从 global 搬到 shared。

`uint128_t` 表示一次 copy atom 以 128 bit 为单位搬运，也就是对 half 来说一次搬 8 个 half。

### `g2s_tiled_copy.get_slice(idx)`

代码：

```cpp
auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
```

取当前线程负责的 copy slice。

### `partition_S(src_tensor)`

代码：

```cpp
auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
```

把 source tensor 按 tiled copy 的 source 布局切成当前线程要读的片段。

对 A：

```text
tAgA_copy : (CPY, CPY_M, CPY_K, k)
          : ((_8,_1), _4, _1, K/kTileK)
```

含义：

```text
CPY   = 每线程每次 copy 的基本数据形状，这里 product 为 8 个 half
CPY_M = M 方向需要重复的 copy tile 数
CPY_K = K 方向需要重复的 copy tile 数
k     = K 方向共有多少个 kTileK tile
```

### `partition_D(dst_tensor)`

代码：

```cpp
auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);
```

把 destination tensor 按 tiled copy 的 destination 布局切成当前线程要写的片段。

对 A：

```text
tAsA_copy : (CPY, CPY_M, CPY_K, kStage)
          : ((_8,_1), _4, _1, _3)
```

source 和 destination 的前几个逻辑维度对应同一次 copy，最后一维变成 shared memory 的 pipeline stage。

### `copy(g2s_tiled_copy, src, dst)`

代码：

```cpp
copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
     tAsA_copy(_, _, _, istage));
```

执行 tiled copy。因为 `g2s_tiled_copy_a` 的 copy atom 是 `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>`，这里会发出 `cp.async` 风格的异步 global-to-shared copy。

`(_, _, _, istage)` 表示：

```text
保留 CPY/CPY_M/CPY_K 三个维度
固定第 istage 个 K tile 或 shared stage
```

注意这里使用了 unqualified `copy`，因为文件开头有：

```cpp
using namespace cute;
```

它等价于使用 CUTE 的 copy。

### `cp_async_fence()`

代码：

```cpp
cp_async_fence();
```

提交当前线程此前发出的 `cp.async` copy group。

可以把它理解为：

```text
前面这些 cp.async 属于同一批，请把这一批提交出去
```

它不等于等待完成，只是 fence/commit。

### `cp_async_wait<N>()`

代码：

```cpp
cp_async_wait<kStage - 2>();
```

等待异步 copy pipeline 中只剩下 `N` 组未完成。

在 `kStage = 3` 时：

```cpp
cp_async_wait<1>();
```

表示等到足够早的一批 global-to-shared 数据已经可读，同时允许后面还有一批在路上。这样可以做到加载和计算重叠。

## 7. S2R：shared memory 到 register

### `make_tiled_copy_A(copy_atom, tiled_mma)`

代码：

```cpp
auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
```

根据 `tiled_mma` 的 A operand 布局，生成一个从 shared memory 搬到 A register fragment 的 tiled copy。

这里的 copy atom 是：

```cpp
SM75_U32x4_LDSM_N
```

也就是 `ldmatrix` 路径，从 shared memory 读取矩阵片段到 register，供 tensor core MMA 使用。

`make_tiled_copy_A` 的关键作用是：让 copy 的目标布局和 `thr_mma.partition_fragment_A` 得到的 register fragment 布局一致。

### `make_tiled_copy_B(copy_atom, tiled_mma)`

代码：

```cpp
auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
```

和 A 类似，但针对 B operand 的 layout。

### `s2r_tiled_copy.get_slice(idx)`

代码：

```cpp
auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
```

取当前线程在这个 shared-to-register copy 中负责的部分。

### `partition_S(sA)`

代码：

```cpp
auto tAsA = s2r_thr_copy_a.partition_S(sA);
```

把 shared memory tensor `sA` 按照 `ldmatrix` 需要的 source layout 切分。

典型 shape：

```text
tAsA : (CPY, CPY_M, CPY_K, kStage)
     : ((_8,_1), _4, _2, _3)
```

这里 `CPY_K = 2` 来自：

```text
kTileK / MMA_K_tile = 32 / 16 = 2
```

### `retile_D(tCrA)`

代码：

```cpp
auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);
```

把已经存在的 register fragment `tCrA` 重新解释成 tiled copy 的 destination 视图。

它不分配新 register，也不移动数据，只是换一个 view：

```text
tCrA 原本按 MMA fragment 维度组织
tCrA_view 按 copy destination 维度组织
```

典型 shape：

```text
tCrA_view : (CPY, CPY_M, CPY_K)
           : ((_8,_1), _4, _2)
```

这样下面的 copy 就能直接写入 `tCrA` 对应的 register：

```cpp
copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read),
     tCrA_view(_, _, ik));
```

## 8. Tensor 切片语法 `tensor(_, ..., idx)`

代码中大量出现：

```cpp
gA(_, _, 0)
tAsA(_, _, ik, ismem_read)
tCrA_view(_, _, ik)
```

CUTE tensor 支持类似函数调用的 slicing。

规则：

```text
_     = 保留该维度
整数  = 固定该维度，取一个 slice
```

例如：

```cpp
gA(_, _, 0)
```

表示取第 0 个 K tile，保留 tile 内的 M/K 维度：

```text
gA : (128, 32, k_tile_count)
gA(_, _, 0) : (128, 32)
```

再如：

```cpp
tAsA(_, _, ik, ismem_read)
```

表示：

```text
保留 CPY、CPY_M
固定 CPY_K = ik
固定 shared stage = ismem_read
```

## 9. `size<dim>(tensor)`

代码：

```cpp
int nk = size<2>(tCrA);
```

返回 tensor 第 `dim` 个 mode 的元素数量。

这里：

```text
tCrA : (MMA, MMA_M, MMA_K)
```

所以：

```cpp
size<2>(tCrA) = MMA_K = 2
```

kernel 里用它来遍历一个 `kTileK = 32` tile 内部的两个 `K = 16` MMA 子步。

后面 epilogue 中：

```cpp
int step = size<3>(tCsC_r2s);
```

表示 C shared scratch 一次能容纳多少个 pipe/batch。

## 10. 主循环里的 `cute::copy`

### S2R 预取下一片

代码：

```cpp
cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
           tCrA_view(_, _, ik_next));
```

从 shared memory 当前读 stage 里，把下一次 MMA 要用的 A fragment 搬到 register。

B 同理：

```cpp
cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
           tCrB_view(_, _, ik_next));
```

这里是同步 shared-to-register load，一般对应 `ldmatrix`。

### G2S 预取下一 K tile

代码：

```cpp
cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
           tAsA_copy(_, _, _, ismem_write));
```

如果还有后续 K tile，就用 `cp.async` 把下一块 A/B 从 global 搬到 shared 的写 stage。

这部分和当前正在进行的 MMA 计算重叠。

## 11. `cute::gemm`

代码：

```cpp
cute::gemm(tiled_mma, tCrD,
           tCrA(_, _, ik),
           tCrB(_, _, ik),
           tCrD);
```

执行一次 tiled MMA：

```text
D = A * B + D
```

输入输出含义：

```text
tiled_mma       = 描述如何调用底层 tensor core MMA atom
tCrD            = accumulator，既是输入 C 又是输出 D
tCrA(_,_,ik)    = 当前 K 子步的 A register fragment
tCrB(_,_,ik)    = 当前 K 子步的 B register fragment
tCrD            = 写回 accumulator fragment
```

这行会在每个 `itile` 的每个 `ik` 上累加：

```text
for K tile:
  for K tile 内部的 MMA_K:
    D += A_sub * B_sub
```

## 12. C epilogue：register -> shared -> global

这个 kernel 没有直接：

```cpp
cute::copy(tCrD, tCgD);
```

而是用 shared memory 做 scratchpad：

```text
register accumulator -> shared memory -> global memory
```

目的是让 global store 使用更宽的 `uint128_t` 向量化写回。

### `sA(_, _, ismem_read).data()`

代码：

```cpp
auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});
```

`sA(_, _, ismem_read)` 取 shared A 当前某个 stage 的 slice，`.data()` 取这个 slice 的底层 pointer。

然后用同一块 shared memory 重新包装成 `sC`：

```text
原来这块 shared 用作 A stage
计算结束后复用为 C 写回 scratchpad
```

这里没有分配新 shared memory，只是换了 layout。

### `make_tiled_copy_C(copy_atom, tiled_mma)`

代码：

```cpp
auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
```

根据 `tiled_mma` 的 C accumulator layout，生成 register-to-shared 的 copy layout。

当前：

```cpp
R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>
```

`T = half_t`，一个 `int` 是 32 bit，可以装 2 个 half。所以 R2S copy 的基本粒度是：

```text
2 个 half
```

### `retile_S(tCrD)`

代码：

```cpp
auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
```

把 accumulator register fragment `tCrD` 重新解释成 register-to-shared copy 的 source 视图。

典型 shape：

```text
tCrC_r2s : (CPY, CPY_M, CPY_N)
          : ((_2,(_2,_2)), _4, _4)
```

解释：

```text
CPY   = 每线程在一个 32x32 C 子块内持有的 8 个 half，
        被 UniversalCopy<int> 按 2 个 half 一组重新分块
CPY_M = CTA 的 M 方向有 4 个 32x32 C 子块
CPY_N = CTA 的 N 方向有 4 个 32x32 C 子块
```

`((_2,(_2,_2)))` 的 product 是 8。

### `partition_D(sC)`

代码：

```cpp
auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);
```

把 shared scratchpad `sC` 按 R2S copy 的 destination 布局切开。

典型 shape：

```text
tCsC_r2s : (CPY, _1, _1, pipe)
```

`sC` 的逻辑形状是：

```text
(32, 32, kSmemLayoutCBatch)
```

它一次只容纳若干个 `32x32` C 子块，所以 M/N 方向对于一个 32x32 tiled copy 来说都是 `_1`，最后一维 `pipe` 表示 C scratchpad 可以暂存几个这样的子块。

### `S2GCopyC`

代码：

```cpp
S2GCopyC s2g_tiled_copy_c;
```

`S2GCopyC` 来自：

```cpp
Copy_Atom<UniversalCopy<cute::uint128_t>, T>
```

这是普通 shared-to-global vectorized copy。`uint128_t` 表示一次搬 128 bit：

```text
128 bit / 16 bit = 8 个 half
```

所以 S2G 的 `CPY` 常见为：

```text
((_8,_1))
```

### `get_thread_slice(idx)`

代码：

```cpp
auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
```

和 `get_slice(idx)` 类似，取当前线程负责的 S2G copy 片段。

### `partition_S(sC)`

代码：

```cpp
auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
```

把 shared scratchpad 作为 S2G 的 source 切分。

典型 shape：

```text
tCsC_s2g : (CPY, _1, _1, pipe)
          : ((_8,_1), _1, _1, pipe)
```

### `partition_D(gD)`

代码：

```cpp
auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);
```

把 global D tile 作为 S2G 的 destination 切分。

典型 shape：

```text
tCgC_s2g : (CPY, CPY_M, CPY_N)
          : ((_8,_1), _4, _4)
```

这里 `CPY_M` 和 `CPY_N` 来自：

```text
128 x 128 的 D tile 被拆成 4 x 4 个 32x32 C 子块
```

### `group_modes<begin, end>(tensor)`

代码：

```cpp
auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);
```

把 tensor 的多个 mode 合并成一个 mode，便于用一个循环线性遍历。

原来：

```text
tCgC_s2g  : (CPY, CPY_M, CPY_N)
tCrC_r2s  : (CPY, CPY_M, CPY_N)
```

合并 mode `[1, 3)`，也就是合并第 1 和第 2 维：

```text
tCgC_s2gx : (CPY, CPY_MN)
tCrC_r2sx : (CPY, CPY_MN)
```

典型打印：

```text
tCgC_s2gx : ((_8,_1), (_4,_4))
tCrC_r2sx : ((_2,(_2,_2)), (_4,_4))
```

这样后面可以：

```cpp
for (int i = 0; i < size<1>(tCrC_r2sx); i += step)
```

把 `4 x 4 = 16` 个 C 子块按线性批次处理。

### R2S copy

代码：

```cpp
cute::copy(r2s_tiled_copy_c,
           tCrC_r2sx(_, i + j),
           tCsC_r2s(_, 0, 0, j));
```

把第 `i + j` 个 C 子块从 register accumulator 写到 shared scratchpad 的第 `j` 个 pipe。

这里：

```text
tCrC_r2sx(_, i + j)  = 某个 32x32 C 子块的 register fragment
tCsC_r2s(_, 0,0,j)  = shared scratchpad 的第 j 个 pipe
```

### S2G copy

代码：

```cpp
cute::copy(s2g_tiled_copy_c,
           tCsC_s2g(_, 0, 0, j),
           tCgC_s2gx(_, i + j));
```

把 shared scratchpad 第 `j` 个 pipe 的 C 子块，以 `uint128_t` 向量化方式写回 global D。

两次 `__syncthreads()` 的意义：

```text
R2S 后同步：确保所有线程都把 register C 写入 shared
S2G 后同步：确保所有线程都完成从 shared 读出，再复用下一批 pipe
```

## 13. 这个 kernel 的 CUTE 数据流总结

整体数据流：

```text
global A/B
  -> make_tensor + local_tile
  -> gA/gB
  -> G2S tiled copy: partition_S/partition_D/copy
  -> shared sA/sB
  -> S2R tiled copy: partition_S/retile_D/copy
  -> register tCrA/tCrB
  -> cute::gemm
  -> register accumulator tCrD
  -> R2S tiled copy: retile_S/partition_D/copy
  -> shared scratch sC
  -> S2G tiled copy: partition_S/partition_D/copy
  -> global D
```

如果只记几个核心概念：

```text
make_tensor       = pointer + layout，构造 view
local_tile        = 从大矩阵切出 CTA tile
partition_S/D     = 按 copy layout 切 source/destination
partition_fragment= 按 MMA layout 创建当前线程 register fragment
retile_S/D        = 不搬数据，只把已有 fragment 换成 copy 视图
copy              = 根据 copy atom 执行实际搬运
gemm              = 根据 tiled_mma 执行 tensor core MMA
group_modes       = 合并维度，方便线性遍历
size              = 查询某一维大小
cosize            = 查询 layout 需要的底层存储容量
```

`gemm_opt_final` 的优化重点是：

```text
1. global -> shared 使用 cp.async 多 stage pipeline
2. shared -> register 使用 ldmatrix 匹配 tensor core MMA
3. accumulator 不直接散写 global，而是 register -> shared -> uint128 global store
```
