# FlashAttention 里的 left_inverse + compose 小例子

这份文档只讲一个问题：

```text
第一次 GEMM 的 C accumulator，
为什么可以通过 left_inverse + compose，
变成第二次 GEMM 的 A operand 来用？
```

核心结论先放前面：

```text
A 坐标
-> trC_as_A_layout
-> 公共 offset
-> left_inverse(trC_as_C_layout)
-> C coord-id
-> trC_mma
-> 真实寄存器位置
```

也就是说：

```text
left_inverse(trC_as_C_layout)
```

就是把：

```text
公共 offset -> C 的 coord-id
```

## 1. 原始代码在干嘛

FlashAttention 里类似这样的代码：

```cpp
cute::gemm(tiled_mma, trC_mma, trA_mma, trB_mma, trC_mma);

auto C_tensor_for_partition = make_tensor_like(gC(_, _));
auto trC_as_A_layout = thr_mma.partition_A(C_tensor_for_partition).layout();
auto trC_as_C_layout = thr_mma.partition_C(C_tensor_for_partition).layout();

auto acc_layout_inv = left_inverse(trC_as_C_layout);
auto a_layout_algebra = acc_layout_inv.compose(trC_as_A_layout);

auto trC_as_A_mma = trC_mma.compose(a_layout_algebra);

cute::gemm(tiled_mma, trD_mma, trC_as_A_mma, tBrB_mma, trD_mma);
```

第一次 GEMM 后，结果在：

```text
trC_mma
```

里面。

但是第二次 GEMM 想把这坨结果当成 A operand 使用。

问题是：

```text
trC_mma 是 C fragment 的坐标形状
第二次 GEMM 需要 A fragment 的坐标形状
```

比如：

```text
C 视角:
    ((2, 2), MMA_M, MMA_N)

A 视角:
    ((2, 2, 2), MMA_M, MMA_N / 2)
```

元素数量一样，但是坐标拆法不一样。

所以目标不是搬数据，而是造一个新的 view：

```text
trC_as_A_mma
```

它满足：

```text
外面用 A 坐标访问，
里面实际读 trC_mma 的 C accumulator 寄存器。
```

## 2. 缩小成 8 个元素

为了能手算，把问题缩小成 8 个元素。

真实寄存器数据假设是：

```text
reg offset:  0  1  2  3  4  5  6  7
data:       d0 d1 d2 d3 d4 d5 d6 d7
```

C 视角坐标写成：

```text
C 坐标 = ((r,c), n)
```

其中：

```text
r = 0/1
c = 0/1
n = 0/1
```

A 视角坐标写成：

```text
A 坐标 = ((r,n,c), q)
```

这里 `q` 只有 1 个值，所以先忽略它。

## 3. C 视角: trC_as_C_layout

设：

```text
trC_as_C_layout = ((2,2),2):((1,4),2)
```

它表示：

```text
C 坐标 ((r,c),n) -> 公共 offset
```

公式是：

```text
offset = r + 4*c + 2*n
```

完整表：

```text
C 坐标        公共 offset
((0,0),0)     0
((1,0),0)     1
((0,0),1)     2
((1,0),1)     3
((0,1),0)     4
((1,1),0)     5
((0,1),1)     6
((1,1),1)     7
```

注意这里的 offset 是公共 offset。

它不是某个普通数组随便的下标，而是 `C_tensor_for_partition` 这个公共 tensor 里的位置。

## 4. C 坐标自己的 coord-id

C shape 是：

```text
((2,2),2)
```

所以 C 坐标的 coord-id 是：

```text
C coord-id = r + 2*c + 4*n
```

完整表：

```text
C 坐标        C coord-id
((0,0),0)     0
((1,0),0)     1
((0,1),0)     2
((1,1),0)     3
((0,0),1)     4
((1,0),1)     5
((0,1),1)     6
((1,1),1)     7
```

把 C 的公共 offset 和 C coord-id 放一起：

```text
C 坐标        公共 offset   C coord-id
((0,0),0)     0             0
((1,0),0)     1             1
((0,0),1)     2             4
((1,0),1)     3             5
((0,1),0)     4             2
((1,1),0)     5             3
((0,1),1)     6             6
((1,1),1)     7             7
```

所以：

```text
left_inverse(trC_as_C_layout)
```

就是：

```text
公共 offset -> C coord-id
```

结果表：

```text
公共 offset:  0  1  2  3  4  5  6  7
C coord-id:   0  1  4  5  2  3  6  7
```

也就是：

```text
acc_layout_inv(0) = 0
acc_layout_inv(1) = 1
acc_layout_inv(2) = 4
acc_layout_inv(3) = 5
acc_layout_inv(4) = 2
acc_layout_inv(5) = 3
acc_layout_inv(6) = 6
acc_layout_inv(7) = 7
```

## 5. A 视角: trC_as_A_layout

现在看 A 视角。

设：

```text
trC_as_A_layout = ((2,2,2),1):((1,2,4),0)
```

它表示：

```text
A 坐标 ((r,n,c),q) -> 同一个公共 offset
```

因为 `q` 只有 1 个值，所以公式简化成：

```text
offset = r + 2*n + 4*c
```

完整表：

```text
A 坐标          公共 offset
((0,0,0),0)     0
((1,0,0),0)     1
((0,1,0),0)     2
((1,1,0),0)     3
((0,0,1),0)     4
((1,0,1),0)     5
((0,1,1),0)     6
((1,1,1),0)     7
```

现在关键来了：

```text
trC_as_A_layout 和 trC_as_C_layout 的输出都是公共 offset。
```

所以 A 和 C 可以通过这个 offset 对齐。

## 6. A 坐标怎么找到 C coord-id

代码：

```cpp
auto acc_layout_inv = left_inverse(trC_as_C_layout);
auto a_layout_algebra = acc_layout_inv.compose(trC_as_A_layout);
```

等价于：

```text
a_layout_algebra(A坐标)
    = acc_layout_inv(trC_as_A_layout(A坐标))
```

也就是：

```text
A 坐标
-> 公共 offset
-> C coord-id
```

完整表：

```text
A 坐标          A 算出的公共 offset   acc_layout_inv(offset)   C coord-id
((0,0,0),0)     0                    0                        0
((1,0,0),0)     1                    1                        1
((0,1,0),0)     2                    4                        4
((1,1,0),0)     3                    5                        5
((0,0,1),0)     4                    2                        2
((1,0,1),0)     5                    3                        3
((0,1,1),0)     6                    6                        6
((1,1,1),0)     7                    7                        7
```

所以：

```text
a_layout_algebra:
    A 坐标 -> C coord-id
```

这就是最重要的翻译器。

## 7. 单点完整计算

拿一个 A 坐标：

```text
A 坐标 = ((0,1,0),0)
```

第一步，用 A layout 算公共 offset：

```text
trC_as_A_layout((0,1,0),0)
    = r + 2*n + 4*c
    = 0 + 2*1 + 4*0
    = 2
```

所以：

```text
A 坐标 ((0,1,0),0) -> 公共 offset 2
```

第二步，把公共 offset `2` 喂给 C 的 inverse：

```text
acc_layout_inv(2) = 4
```

所以：

```text
公共 offset 2 -> C coord-id 4
```

第三步，把 C coord-id `4` 解回 C 坐标。

C shape 是：

```text
((2,2),2)
```

C coord-id 公式是：

```text
coord-id = r + 2*c + 4*n
```

现在：

```text
4 = r + 2*c + 4*n
```

能解出：

```text
n = 1
c = 0
r = 0
```

所以：

```text
C coord-id 4 -> C 坐标 ((0,0),1)
```

完整链路：

```text
A 坐标 ((0,1,0),0)
-> 公共 offset 2
-> C coord-id 4
-> C 坐标 ((0,0),1)
```

这说明：

```text
A 视角里的 ((0,1,0),0)
对应 C 视角里的 ((0,0),1)
```

## 8. 为什么不直接用 offset 读 C

因为 `trC_mma` 不是裸数组接口。

它是带 layout 的 tensor view。

它的入口是：

```text
C coord-id / C 坐标
```

不是：

```text
公共 offset
```

所以从 A 得到公共 offset 后，还差一步：

```text
公共 offset -> C coord-id
```

这一步就是：

```cpp
left_inverse(trC_as_C_layout)
```

换句话说，你脑子里说的：

```text
offset 算 C coord-id，然后去 C 里找
```

在 CuTe 里正式写法就是：

```cpp
left_inverse(trC_as_C_layout)
```

## 9. 最后 compose 到 trC_mma

现在已经有：

```text
a_layout_algebra:
    A 坐标 -> C coord-id
```

原来的：

```text
trC_mma:
    C coord-id -> 真实寄存器 offset
```

所以：

```cpp
auto trC_as_A_mma = trC_mma.compose(a_layout_algebra);
```

就是：

```text
trC_as_A_mma:
    A 坐标 -> C coord-id -> 真实寄存器 offset
```

也就是说：

```text
trC_as_A_mma 外表像 A operand，
但里面读的是 trC_mma 的 C accumulator 寄存器。
```

这就是第二次 GEMM 可以写成：

```cpp
cute::gemm(tiled_mma, trD_mma, trC_as_A_mma, tBrB_mma, trD_mma);
```

## 10. 一句话总结

FlashAttention 这里的 `left_inverse + compose` 不是在搬数据。

它是在造一个坐标翻译器：

```text
A fragment 坐标 -> C fragment coord-id
```

完整链路：

```text
A 坐标
-> A layout 算公共 offset
-> C layout 的 inverse 把公共 offset 变成 C coord-id
-> trC_mma 用 C coord-id 读原来的 accumulator 寄存器
```

所以：

```text
left_inverse 的作用就是：
    从公共 offset 反查到目标 layout 的 coord-id。
```
