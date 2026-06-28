# GEMM 8192 性能异常：Cache Set Conflict 与 Partition Camping

这份笔记整理的是 `gemm-opt.cu` 里手写 half GEMM kernel 在方阵 sweep benchmark 中遇到的一个现象：

```text
1024..7168 大多能达到 cuBLASLt best heuristic candidates 的 90% 到 100%+
3072 附近有时能稳定领先
8192 明显掉到六七成左右
```

更关键的是，8192 的邻近尺寸并不会同样掉：

```text
8064 正常
8192 明显掉
8320 又恢复
```

所以这个问题不像是“矩阵越大越慢”，更像是 `8192` 这个 power-of-two leading dimension 触发了全局内存地址映射上的冲突。

## 面试版一句话

`8192` 对 half 矩阵来说意味着 row stride 是 `8192 * 2 = 16384 bytes = 16KB = 2^14`。这种大 2 的幂 stride 会让不同 row/tile 的地址 bit 模式高度重复。在 set-associative cache 和 GPU memory partition 这种按地址 bit 映射的硬件结构里，重复的地址模式可能让大量不同 cache line 集中落到少数 cache set、L2 slice 或 memory partition 上，造成 conflict miss 或 partition camping。邻近的 `8064/8320` 因为 stride 不那么整齐，地址映射被打散，所以性能恢复。

## 现象

当前 compare 模式已经把 cuBLASLt 改成了“遍历 heuristic 返回的所有候选 algo，实测取最快”，所以比较对象不是 `algo 0`，而是当前候选中的 best。

典型结果形状如下：

```text
SIZE   achieve vs cuBLASLt best
1024   90%+
2048   90%+
3072   100%+
4096   接近打平
5120   90%+
6144   90%+
7168   90%+
8192   明显掉到 60%-70%+
```

邻近点测试更能说明问题：

```text
7680 正常
7808 正常
7936 正常
8064 正常
8192 异常下跌
8320 恢复
8448 基本恢复
```

因此 8192 是一个特殊坏点，而不是大矩阵普遍退化。

## Cache 基础：Set Associative Mapping

Cache 通常不是一个“任意位置都能放”的大盒子。为了硬件查找快，它会被分成多个 set，每个 set 里面有若干个 way：

```text
cache
  set 0: way 0, way 1, way 2, ...
  set 1: way 0, way 1, way 2, ...
  set 2: way 0, way 1, way 2, ...
  ...
```

一个地址大致会被拆成：

```text
address = [ tag | set index | line offset ]
```

访问流程是：

```text
1. 根据 set index 找到一个 cache set
2. 在这个 set 的多个 way 里比较 tag
3. tag 命中就是 cache hit
4. tag 都不匹配就是 miss
5. 如果 set 已经满了，新 line 会踢掉旧 line
```

重点是：一个 cache line 不能放到任意 set，它只能放进地址映射到的那个 set。

如果很多不同 cache line 都映射到同一个 set，并且数量超过这个 set 的 way 数，就会互相驱逐：

```text
4-way set:

访问 A B C D 可以放下
访问 A B C D E 就需要踢掉一个
再访问 A 时可能又 miss
```

这就是 conflict miss。它不是 cache 总容量不够，而是某些 set 被打爆，其他 set 可能还没被充分利用。

## 为什么 8192 容易触发

以 half 矩阵为例，一个元素是 2 bytes。

当 `K = 8192` 时，一行 A 的跨度是：

```text
8192 * 2 = 16384 bytes = 16KB = 2^14
```

如果假设 cache line 是 128B，那么一行跨度等于：

```text
16384 / 128 = 128 条 cache line
```

用一个简化模型说明。假设 cache 有 256 个 set，set index 粗略是：

```text
set_id = (address / 128) % 256
```

每跨一行，set id 增加 128：

```text
0, 128, 0, 128, 0, 128, ...
```

也就是说，访问模式可能只在少数 set 之间循环。

而如果 `K = 8064`：

```text
8064 * 2 = 16128 bytes
16128 / 128 = 126 条 cache line
```

每跨一行，set id 增加 126：

```text
0, 126, 252, 122, 248, 118, ...
```

这个序列会扫过更多 set，分布更散。

更一般地说，容易冲突的程度和下面这个值有关：

```text
gcd(stride_in_cache_lines, number_of_sets)
```

如果最大公约数很大，访问只能落到少数 set；如果最大公约数小，访问能分散到更多 set。

在上面的简化模型里：

```text
8192: stride = 128 cache lines
gcd(128, 256) = 128
只能访问 256 / 128 = 2 个 set

8064: stride = 126 cache lines
gcd(126, 256) = 2
能访问 256 / 2 = 128 个 set
```

真实 GPU 的 L2 映射不会这么简单，通常会有 hash 或 XOR，但 power-of-two stride 仍然更容易制造周期性地址模式。

## 为什么 tile 固定也会出问题

手写 GEMM 的 tile 形状可能是固定的，比如：

```text
128 x 128 x 32
```

但 tile 在 global memory 里的地址位置由 leading dimension 决定。

A 的地址大致是：

```cpp
A[row * K + col]
```

相邻行的地址差是：

```text
K * sizeof(half)
```

相邻 M tile 的 A 起点差是：

```text
128 * K * sizeof(half)
```

当 `K = 8192`：

```text
128 * 8192 * 2 = 2MB
```

这也是一个很整齐的 2 的幂。也就是说，不只是行内访问，tile-to-tile 的起点地址也会表现出强周期性。

所以“tile 固定”只说明每个 CTA 搬运和计算的形状固定；真正决定 cache set、L2 slice、memory partition 分布的是 global address pattern，而这个 pattern 由 `M/N/K` 的 leading dimension 决定。

## Cache 命中和冲突不是一回事

如果多个 CTA 访问同一份数据，这是 cache reuse：

```text
block(m0, n0) 读 B tile X
block(m1, n0) 也读 B tile X
```

这种情况下 cache 命中率会提高。

但 8192 问题更像是：

```text
block 0 读 B tile X -> set 0
block 1 读 B tile Y -> set 0
block 2 读 B tile Z -> set 0
```

这些是不同数据，只是映射到了同一批 set 或 partition。它们不会互相复用，反而会互相驱逐或排队。

所以不是“踩同一个地方所以命中更高”，而是“不同数据挤进同一个入口”。

## GPU 上还要考虑 Memory Partition

GPU 的 global memory 带宽来自多个 memory partition / L2 slice 并行工作。

理想情况：

```text
partition 0 忙
partition 1 忙
partition 2 忙
partition 3 忙
...
```

坏情况：

```text
partition 0 爆满
partition 1 很闲
partition 2 很闲
...
```

这类问题常叫 partition camping。

8192 这种 2 的幂 stride 可能让不同 CTA 访问的不同地址落到相同或少数几个 partition 上，导致显存总带宽没有被均匀吃满。

因此，8192 的异常可能同时包含：

```text
1. cache set conflict / conflict miss
2. L2 slice 映射不均
3. memory partition camping
4. CTA 调度顺序和地址相位共振
```

## 为什么 cuBLASLt 不容易掉

cuBLASLt 不是一个单一 GEMM kernel，而是一组 kernel 和调度策略：

```text
不同 tile shape
不同 pipeline stage
不同 CTA swizzle
不同 split-K / stream-K
不同 epilogue
不同 workspace 策略
针对特殊 shape 的 kernel selection
```

对于 8192 这种 power-of-two shape，cuBLASLt 可能会通过更复杂的 tile swizzle、kernel selection、内部 padding 或 split-K 策略避开地址映射共振。

手写 kernel 如果只有一个固定 config，就很容易在某些特殊 size 上踩坑。

## 如何验证

可以用 benchmark 先验证现象：

```bash
./gemm-opt compare 7680 8448 128 50 10
```

参数含义：

```text
compare start end step count warmup

7680: 起始 size
8448: 结束 size
128 : 每次递增
50  : 正式计时重复次数
10  : warmup 次数
```

如果只有 8192 明显掉，而 8064、8320 恢复，就支持“特殊 stride/address mapping 问题”的判断。

进一步可以用 Nsight Compute 看这些指标：

```text
L2 hit rate
L2 sector miss
DRAM throughput
memory partition utilization
warp stall memory dependency
long scoreboard stalls
```

如果 8192 相比邻近点出现 L2 miss 增加、DRAM partition 利用不均、memory stall 上升，就能更强地证明这个方向。

## 缓解思路

常见做法：

```text
1. CTA swizzle
   改变 block 访问 tile 的顺序，让不同 CTA 的访问相位错开。

2. Padding leading dimension
   把 8192 pad 成 8192 + small offset，打破 power-of-two stride。

3. 换 tile shape
   改变 tile-to-tile 的地址步长和 CTA 访问节奏。

4. split-K / stream-K
   改变 K 维任务分解和 L2 reuse pattern。

5. 多 config dispatch
   对 8192 这种特殊 shape 单独选 kernel，其他 shape 用普通 kernel。

6. fallback cuBLASLt
   如果目标是实际应用吞吐，可以只在自己稳定领先的 shape 上用自写 kernel。
```

当前代码里已经尝试过一种简单 CTA swizzle：

```cpp
iy = blockIdx.x / swizzle_n;
ix = blockIdx.y * swizzle_n + ((blockIdx.x + iy) % swizzle_n);
```

它的目标是让不同 `iy` 的 CTA 在 `ix` 方向错开，减少所有 CTA 同相位访问同一批地址模式。不过这个方法只能缓解，不能保证完全消除 8192 问题。

## 面试时可以这样讲

可以按这个顺序回答：

```text
1. 我发现 8192 方阵 GEMM 明显掉性能，但邻近的 8064/8320 正常。

2. 因为 half 下 8192 的 row stride 是 16KB，也就是 2^14 bytes，地址模式高度周期化。

3. Cache 是 set-associative 的，地址只能映射到某个 set 的有限 ways；GPU 还有 L2 slice 和 memory partition 映射。

4. Power-of-two stride 可能让不同 tile 的不同 cache line 集中落到少数 set 或 partition，导致 conflict miss 或 partition camping。

5. 这不是 cache capacity 不够，也不是计算量变大，而是地址映射不均和 CTA 访问相位共振。

6. 解决方向是 CTA swizzle、padding leading dimension、换 tile shape、split-K/stream-K，或者对特殊 shape 做多 config dispatch。
```

一句更凝练的版本：

```text
8192 的问题本质上是 power-of-two leading dimension 导致 global memory 地址序列周期过短，使 set-associative cache 和 GPU memory partition 的映射集中化。不同数据没有形成 cache reuse，而是竞争同一批 cache set/partition，于是出现 conflict miss 或 partition camping，最终导致 GEMM throughput 在 8192 这个特殊点明显下降。
```

