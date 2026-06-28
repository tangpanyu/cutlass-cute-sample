# CuTe GEMM 总结

这份笔记用 `gemm-opt.cu` / `gemm-opt_copy.cu` 里的 half GEMM 为例，整理 CuTe 写 GEMM 时最容易绕的几个点：MMA、copy、pipeline、epilogue，以及变量命名。

## 整体数据流

一个 threadblock 计算一个 `D` tile：

```text
A global: (M, K)
B global: (N, K)
D global: (M, N)

gA/gB -> sA/sB -> tCrA/tCrB -> MMA accumulate tCrD -> sC -> gD
global -> shared -> register -> register accumulator -> shared -> global
```

数学上是：

```text
D[m, n] = sum_k A[m, k] * B[n, k]
```

代码里 `B` 按 `(N, K)` 存，所以 GEMM 语义等价于 `A @ B^T`。

## CuTe 的基本套路

CuTe 不是直接写“线程 x 读第几个元素”，而是先描述一个大的逻辑 tile，然后用 layout 把它拆成每个线程的任务。

### MMA

MMA 的构成到执行：

1. 选择 MMA atom，例如 `SM80_16x8x16_F16F16F16F16_TN`。
2. 用 `make_tiled_mma` 把 atom 在 warp/threadblock 内铺开。
3. 用 `thr_mma = tiled_mma.get_slice(threadIdx.x)` 取得当前线程视角。
4. 用 `partition_fragment_A/B/C` 创建当前线程的 register fragment。
5. 在 K 方向循环里调用 `cute::gemm(...)` 执行 MMA。

典型代码：

```cpp
TiledMMA tiled_mma;
auto thr_mma = tiled_mma.get_slice(threadIdx.x);

auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)

clear(tCrD);

for (int ik = 0; ik < size<2>(tCrA); ++ik) {
  cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
}
```

`MMA` / `MMA_M` / `MMA_N` / `MMA_K` 不是普通矩阵坐标，更像执行维度：

```text
MMA    = 当前线程在一个 MMA atom 中持有的寄存器 value
MMA_M  = M 方向需要重复几个 tiled MMA
MMA_N  = N 方向需要重复几个 tiled MMA
MMA_K  = K 方向需要重复几个 tiled MMA
```

### Copy

copy 的构成到执行：

1. 选择 copy atom，例如 `cp.async`、`ldmatrix`、`uint128_t` vector copy。
2. 如果 copy layout 独立于 MMA，比如 global -> shared，用 `make_tiled_copy` 显式指定线程和值布局。
3. 如果 copy 是为了喂 MMA，比如 shared -> register，用 `make_tiled_copy_A/B(atom, tiled_mma)` 从 MMA layout 自动派生。
4. 当前线程调用 `get_slice(threadIdx.x)`。
5. 对 source 用 `partition_S`，对 destination 用 `partition_D` 或 `retile_D`。
6. 调用 `copy(tiled_copy, src_view, dst_view)`。

典型 global -> shared：

```cpp
G2SCopyA g2s_tiled_copy_a;
auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);

auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // source: global A
auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // dest: shared A

copy(g2s_tiled_copy_a,
     tAgA_copy(_, _, _, itile_to_read),
     tAsA_copy(_, _, _, ismem_write));
```

典型 shared -> register：

```cpp
auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);

auto tAsA = s2r_thr_copy_a.partition_S(sA);
auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

copy(s2r_tiled_copy_a,
     tAsA(_, _, ik, ismem_read),
     tCrA_view(_, _, ik));
```

## 变量命名

常见命名可以这样拆：

```text
t = thread-level tensor，当前线程看到/负责的 view
g = global memory
s = shared memory
r = register
A/B/C/D = 逻辑矩阵或 operand
g2s = global -> shared
s2r = shared -> register
r2s = register -> shared
s2g = shared -> global
x = group_modes 后的扁平 view
```

例子：

```text
tCrA       当前线程的 MMA register A fragment
tCrB       当前线程的 MMA register B fragment
tCrD       当前线程的 accumulator register fragment
tAgA_copy  g2s copy 中当前线程负责读的 global A view
tAsA_copy  g2s copy 中当前线程负责写的 shared A view
tAsA       s2r copy 中当前线程负责读的 shared A view
tCrA_view  tCrA 的 copy 视角 view，用来让 ldmatrix/copy 写入
tCrC_r2s   C accumulator 的 register -> shared source view
tCsC_s2g   shared C -> global 的 source view
tCgC_s2gx  global C destination view 合并 M/N 后的一维 view
```

`tCrA` 和 `tCrA_view` 不是两份数据。它们是同一批 register fragment 的两个 view：

```text
tCrA       给 MMA 使用，shape 类似 (MMA, MMA_M, MMA_K)
tCrA_view  给 copy 使用，shape 类似 (CPY, CPY_M, CPY_K)
```

## `local_tile` 为什么会多一维

原始 `A` 是二维：

```cpp
Tensor A = make_tensor(...); // (M, K)
```

但：

```cpp
Tensor gA = local_tile(A,
                       make_tile(Int<kTileM>{}, Int<kTileK>{}),
                       make_coord(iy, _)); // (kTileM, kTileK, k)
```

`make_coord(iy, _)` 固定了 M 方向 tile，但 K 方向保留 `_`，所以 `gA` 变成：

```text
gA: (tile内M, tile内K, 第几个K tile)
```

因此：

```cpp
gA(_, _, 0)
```

取的是当前 M tile 下第 0 个 K tile，shape 回到 `(kTileM, kTileK)`，可用于创建 MMA fragment 的 layout。后续真正循环用的是 `itile_to_read` 这类 K tile 编号。

## `partition_*` 和 `retile_*`

### `partition_S` / `partition_D`

`partition_S` 和 `partition_D` 用来把一个 shared/global 这种“大 tensor”按当前 copy layout 切给当前线程：

```cpp
auto src = thr_copy.partition_S(gA);
auto dst = thr_copy.partition_D(sA);
```

它回答的是：

```text
当前线程应该读 source 的哪些元素？
当前线程应该写 destination 的哪些元素？
```

### `retile_S` / `retile_D`

`retile_*` 不重新按线程切分，它只是把一个已经 thread-local 的 fragment 改成 copy 需要的 view。

例如 `tCrA` 已经是当前线程自己的 register fragment，所以不能再 `partition_D(tCrA)`。应该用：

```cpp
auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);
```

可以类比：

```python
buf = torch.empty(...)
mma_view = buf.view(mma_shape)
copy_view = buf.view(copy_shape)
```

## global -> shared 的 cp.async pipeline

多 stage pipeline 的几个状态变量：

```text
itile_to_read = 下一个要从 global 读的 K tile
ismem_read    = 当前要从 shared 读的 stage
ismem_write   = 下一个要写入的 shared stage
```

prologue 先提交 `kStage - 1` 个 stage：

```cpp
for (int istage = 0; istage < kStage - 1; ++istage) {
  copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
       tAsA_copy(_, _, _, istage));
  copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
       tBsB_copy(_, _, _, istage));
  cp_async_fence();
}
```

然后：

```cpp
cp_async_wait<kStage - 2>();
__syncthreads();
```

`cp_async_wait<N>()` 的语义是等待到最多只剩 `N` 个最近的 cp.async group 还可能 pending。比如 `kStage=3` 时 `wait<1>`，表示允许最近 1 个 group 继续在路上，但更老的 group 必须完成。

尾部没有真实 copy 时，代码仍然执行：

```cpp
cp_async_fence();
```

这会提交一个 empty cp.async-group。PTX ISA 规定：没有未提交 cp.async 时，`commit_group` 会产生 empty group；empty group trivially complete。这样可以推进 group 队列，让 `wait_group` 的节奏在 tail 阶段仍然成立。

### 为什么在 `ik == 0` 发起下一次 gmem -> smem

这份代码的 pipeline 相位是：

```text
prologue 只预加载 kStage - 1 个 stage
正文每个 K tile 开头，也就是 ik == 0 时，把剩下那个空 stage 补上
```

以 `kStage=3`、每个 K tile 内 `nk=2` 为例，可以看到这个相位的空隙：

```text
prologue:
  load tile0 -> stage0
  load tile1 -> stage1
  stage2 空

tile0, ik=0:
  load tile2 -> stage2
  compute tile0 ik0

tile0, ik=1:
  wait stage1 ready
  read 切到 stage1
  compute tile0 ik1
  stage0 已经用完，但此时不马上加载

tile1, ik=0:
  load tile3 -> stage0
  compute tile1 ik0
```

所以 `ik == nk - 1` 时，当前 stage 的数据确实已经读进寄存器或者快算完了，那个 stage 可以复用。但当前代码没有在这个点马上发射新的 global load，而是等到下一个 tile 的 `ik == 0`。这不是 correctness 问题，而是 pipeline 相位选择。

但要注意：当 `nk=2` 时，这个相位并不理想。`stage0` 在 `tile0, ik=1` 附近已经释放，却要等到 `tile1, ik=0` 才重新发射 load，中间少了一段隐藏 global memory latency 的时间。换句话说，虽然配置了 `kStage=3`，但运行中更像只有 `kStage-1` 个 stage 持续有效，第三个 stage 经常处在空档里。

如果希望这种“`ik == 0` 发射”的写法更自然，`nk=3` 这类每个 K tile 内有更多 MMA 子步的情况更合适：从 `ik==0` 发起 gmem -> smem 后，还有更多 `ik` 计算可以覆盖这次异步拷贝。`nk` 越小，越容易暴露这个相位的空隙。

可以把它理解成两种合法写法：

```text
方案 A：prologue 加载 kStage - 1 个，正文在 ik == 0 发
  先故意留一个空 stage
  每个 tile 开头补上这个空 stage

方案 B：prologue 加载 kStage 个，正文在 ik == nk - 1 发
  一开始 stage 全满
  每当一个 stage 用完，马上复用它加载未来 tile
```

当前代码属于方案 A。它的特点是主循环逻辑统一：`ik == 0` 时总有一个 `ismem_write` stage 可以写，并且 prologue 少提交一组 cp.async。方案 B 也能写，可能更早复用刚释放的 stage，但要整体调整 prologue 和发射位置，否则会和 `ik == 0` 的发射冲突。

所以如果 `nk=2` 且 kernel 明显受 global memory latency 影响，方案 B 往往更合理：stage 一用完就在 `ik == nk - 1` 发起下一次 load，让所有 stage 尽量保持有效。当前代码在 compute bound 场景里可能看不出这个问题，因为 MMA 计算本身已经足够长，global load 的空隙被计算吞掉了；但从 pipeline 利用率看，它不是最满的写法。

## shared -> register 和 MMA 主循环

进入主循环前先预取 `ik=0` 到 register：

```cpp
int ik = 0;
copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));
```

主循环里通常是：

```text
1. 预取下一份 register fragment
2. 如果需要，发起下一组 global -> shared
3. 对当前 ik 做 MMA
```

核心结构：

```cpp
for (int itile = 0; itile < ntile; ++itile) {
  int nk = size<2>(tCrA);

  for (int ik = 0; ik < nk; ++ik) {
    int ik_next = (ik + 1) % nk;

    if (ik == nk - 1) {
      cp_async_wait<kStage - 2>();
      __syncthreads();
      ismem_read = (ismem_read + 1) % kStage;
    }

    copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
         tCrA_view(_, _, ik_next));
    copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
         tCrB_view(_, _, ik_next));

    if (ik == 0) {
      if (itile_to_read < ntile) {
        copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
             tAsA_copy(_, _, _, ismem_write));
        copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
             tBsB_copy(_, _, _, ismem_write));

        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
      }
      cp_async_fence();
    }

    cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
  }
}
```

这里维度参数更像“第几次 copy/MMA 任务”，不是手写物理地址。真实地址由 tensor layout 映射。

## Epilogue: register -> shared -> global

accumulator `tCrD` 在 register 里。为了合并访存写回 global，通常先写到 shared scratchpad，再从 shared vectorized 写 global：

```text
tCrD -> sC -> gD
```

典型代码：

```cpp
auto sC = make_tensor(sA.data(), SmemLayoutC{});

auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(threadIdx.x);
auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

S2GCopyC s2g_tiled_copy_c;
auto s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(threadIdx.x);
auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);
```

`sC` 常常复用 `sA` 的 shared memory 指针：

```cpp
auto sC = make_tensor(sA.data(), SmemLayoutC{});
```

这里不是沿用 `sA` 的 layout，而是只借用 shared memory 起始指针，并用 `SmemLayoutC` 重新解释这块内存。

## 为什么要 `group_modes`

输出 tile 往往被拆成二维小块，比如：

```text
tCrC_r2s: (CPY, CPY_M, CPY_N)
tCgC_s2g: (CPY, CPY_M, CPY_N)
```

如果 `CPY_M=4`、`CPY_N=4`，就是 16 个 C 小块。`sC` 的 pipe 可能一次只能暂存 4 个小块，所以把 `(CPY_M, CPY_N)` 合成一维更方便：

```cpp
auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // (CPY, CPY_MN)
auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // (CPY, CPY_MN)
```

然后：

```cpp
int step = size<3>(tCsC_r2s); // pipe 数

for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
  for (int j = 0; j < step; ++j) {
    copy(r2s_tiled_copy_c,
         tCrC_r2sx(_, i + j),
         tCsC_r2s(_, 0, 0, j));
  }
  __syncthreads();

  for (int j = 0; j < step; ++j) {
    copy(s2g_tiled_copy_c,
         tCsC_s2g(_, 0, 0, j),
         tCgC_s2gx(_, i + j));
  }
  __syncthreads();
}
```

语义是：

```text
线性遍历所有 C subtile，每次处理 pipe 个 subtile。
```

## 常见坑

### tile 坐标必须初始化

不要只写：

```cpp
int ix, iy;
```

必须用 block id 或 swizzle 算出来：

```cpp
ix = blockIdx.x;
iy = blockIdx.y;
```

或：

```cpp
get_swizzled_tile_coord<Config>(m, n, ix, iy);
if (ix < 0) return;
```

### B 和 D 的 tensor 不要切错

正确：

```cpp
Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
```

### `partition_D` 要从 thread slice 调用

正确：

```cpp
auto thr_copy = tiled_copy.get_slice(threadIdx.x);
auto dst = thr_copy.partition_D(sA);
```

不是：

```cpp
tiled_copy.partition_D(sA);
```

### shared stage 要取模

主循环里：

```cpp
ismem_write = (ismem_write + 1) % kStage;
ismem_read  = (ismem_read + 1) % kStage;
```

否则会访问不存在的 stage。

### `uint128_t` copy atom 需要足够的 value layout

如果使用：

```cpp
Copy_Atom<UniversalCopy<cute::uint128_t>, T>
```

一次 atom 需要 128 bit。对于 half 来说是 8 个 half，所以 `make_tiled_copy` 通常需要类似：

```cpp
make_layout(make_shape(Int<1>{}, Int<8>{}))
```

作为 value layout，否则可能触发：

```text
TiledCopy uses too few vals for selected CopyAtom
```

## 一句话心智模型

CuTe GEMM 不是“手写每个线程访问哪个地址”，而是：

```text
先定义 tile 和 atom，
再用 layout 把 tile 分给线程，
最后通过 partition/retile 得到每个线程的执行 view。
```

读 shape 时也要按执行结构理解：

```text
(CPY, CPY_M, CPY_K, stage) = 当前线程一次 copy 的 value，以及为覆盖 tile 需要重复的次数
(MMA, MMA_M, MMA_N)        = 当前线程一次 MMA atom 的 value，以及为覆盖 tile 需要重复的次数
```
