# CuTe Layout: left_inverse 和 composition 笔记

这份笔记总结 CuTe layout algebra 里 `left_inverse` 和 `composition/compose` 的直觉。

## 1. Layout 的本质

一个 CuTe layout 本质上是一个映射：

```text
逻辑坐标 -> 物理 offset
```

比如：

```cpp
auto layout_a = make_layout(make_shape(Int<4>{}, Int<4>{}),
                            make_stride(Int<4>{}, Int<1>{}));
```

这个 layout 是：

```text
(4,4):(4,1)
```

它表示：

```text
(m,n) -> m * 4 + n
```

也就是一个 4x4 的 row-major layout。

注意：layout 本身不拥有数据，也不搬数据。它只是告诉 tensor：给我一个坐标，我应该访问 pointer 后面的哪个 offset。

## 2. left_inverse: 从零开始看

先不要想 `composition`，只看 `left_inverse`。

一个 layout 做的是：

```text
逻辑坐标 -> 物理 offset
```

`left_inverse(layout)` 做的是：

```text
物理 offset -> 原逻辑坐标的 coord-id
```

注意这里有两个编号，千万不要混：

```text
offset:
    layout 用 stride 算出来的物理偏移。

coord-id:
    CuTe 给 shape 里的坐标规定的线性编号。
```

### 2.1 原 layout

用一个最小例子：

```text
L = (2,3):(3,1)
```

它的 shape 是：

```text
(2,3)
```

坐标写成：

```text
(m,n)
```

其中：

```text
m = 0..1
n = 0..2
```

stride 是：

```text
(3,1)
```

所以 offset 公式是：

```text
L(m,n) = 3*m + n
```

完整表：

```text
原坐标    offset
(0,0)     0
(0,1)     1
(0,2)     2
(1,0)     3
(1,1)     4
(1,2)     5
```

所以普通 layout `L` 是：

```text
原坐标 -> offset
```

比如：

```text
L(0,2) = 2
L(1,0) = 3
L(1,2) = 5
```

### 2.2 CuTe 的 coord-id

CuTe 对 shape `(2,3)` 里的坐标，还有一套默认编号，叫 `coord-id`。

这套编号等价于 identity layout：

```text
E_old = (2,3):(1,2)
```

也就是：

```text
coord-id(m,n) = m + 2*n
```

为什么乘 `2`？因为第 0 维大小是 `2`。

完整表：

```text
原坐标    coord-id
(0,0)     0
(1,0)     1
(0,1)     2
(1,1)     3
(0,2)     4
(1,2)     5
```

所以同一个坐标 `(0,2)` 有两个编号：

```text
L 的 offset:
    L(0,2) = 3*0 + 2 = 2

CuTe 的 coord-id:
    coord-id(0,2) = 0 + 2*2 = 4
```

也就是说：

```text
offset 2   对应原坐标 (0,2)
coord-id 2 对应原坐标 (0,1)
```

这两个 `2` 不是一回事。

### 2.3 left_inverse 要满足什么

`left_inverse(L)` 要构造一个新的 layout，叫它 `Inv`。

它满足：

```text
Inv(L(m,n)) = coord-id(m,n)
```

也就是：

```text
先用 L 把原坐标变成 offset
再用 Inv 把 offset 变回原坐标的 coord-id
```

逐个算：

```text
原坐标 (0,0):
    L(0,0) = 0
    coord-id(0,0) = 0
    所以 Inv(0) = 0

原坐标 (0,1):
    L(0,1) = 1
    coord-id(0,1) = 2
    所以 Inv(1) = 2

原坐标 (0,2):
    L(0,2) = 2
    coord-id(0,2) = 4
    所以 Inv(2) = 4

原坐标 (1,0):
    L(1,0) = 3
    coord-id(1,0) = 1
    所以 Inv(3) = 1

原坐标 (1,1):
    L(1,1) = 4
    coord-id(1,1) = 3
    所以 Inv(4) = 3

原坐标 (1,2):
    L(1,2) = 5
    coord-id(1,2) = 5
    所以 Inv(5) = 5
```

所以 `Inv` 这个函数的结果是：

```text
输入 offset:  0  1  2  3  4  5
Inv 输出:     0  2  4  1  3  5
```

### 2.4 为什么 Inv 的 shape 变成 `(3,2)`

现在问题来了：

```text
Inv = ?
```

CuTe 会把这个反查函数也表示成一个 layout。

原 layout 是：

```text
L = (2,3):(3,1)
```

公式是：

```text
offset = 3*m + n
```

如果从 offset 反解原坐标：

```text
n = offset % 3
m = offset / 3
```

注意顺序：

```text
offset 先解出 n，再解出 m
```

所以 inverse 的输入坐标天然是：

```text
(n,m)
```

而不是：

```text
(m,n)
```

原来：

```text
m 有 2 种
n 有 3 种
```

换成 `(n,m)` 后：

```text
n 有 3 种
m 有 2 种
```

所以 inverse layout 的 shape 是：

```text
(3,2)
```

这就是 `(3,2)` 的来源。

它不是凭空规定的，是从：

```text
offset = 3*m + n
```

反解出来的。

### 2.5 为什么 Inv 的 stride 是 `(2,1)`

`Inv` 的输出要是原坐标的 coord-id。

原坐标 `(m,n)` 的 coord-id 是：

```text
coord-id = m + 2*n
```

但是 `Inv` 的输入坐标顺序是：

```text
(n,m)
```

所以把同一个公式按 `(n,m)` 的顺序重写：

```text
coord-id = 2*n + m
```

这就是一个 layout：

```text
Inv(n,m) = 2*n + 1*m
```

所以：

```text
Inv = (3,2):(2,1)
```

### 2.6 把所有步骤连起来

完整链路是：

```text
原坐标
-> L = (2,3):(3,1)
-> offset
-> offset 按 Inv 的 shape (3,2) 解码成 (n,m)
-> Inv = (3,2):(2,1)
-> 原坐标在 shape (2,3) 里的 coord-id
```

完整表：

```text
原坐标   L 得到 offset   offset 按 Inv.shape=(3,2) 解码   Inv 输出   原 coord-id
(0,0)    0              0 -> (0,0)                    0         0
(0,1)    1              1 -> (1,0)                    2         2
(0,2)    2              2 -> (2,0)                    4         4
(1,0)    3              3 -> (0,1)                    1         1
(1,1)    4              4 -> (1,1)                    3         3
(1,2)    5              5 -> (2,1)                    5         5
```

拿 `(0,2)` 单独看：

```text
原坐标:
    (0,2)

先用 L 算 offset:
    L(0,2) = 2

offset 2 输入给 Inv:
    Inv 的 shape 是 (3,2)
    所以 2 被解码成 Inv 坐标 (2,0)

再用 Inv 的 stride (2,1) 算输出:
    Inv(2,0) = 2*2 + 0 = 4

这个 4 是什么？
    它是原坐标 (0,2) 在 shape (2,3) 里的 coord-id。
```

所以：

```text
Inv(L(0,2)) = 4
```

而：

```text
coord-id(0,2) = 4
```

对上了。

### 2.7 可以怎么记

对 compact row-major 二维 layout：

```text
L = (M,N):(N,1)
```

它的公式是：

```text
offset = N*m + n
```

反解：

```text
n = offset % N
m = offset / N
```

所以 inverse 的输入坐标顺序变成：

```text
(n,m)
```

shape 从：

```text
(M,N)
```

变成：

```text
(N,M)
```

而 inverse 的输出要是原坐标的 coord-id：

```text
coord-id = m + M*n
```

按 `(n,m)` 写就是：

```text
coord-id = M*n + m
```

所以：

```text
left_inverse((M,N):(N,1)) = (N,M):(M,1)
```

例如：

```text
left_inverse((2,3):(3,1)) = (3,2):(2,1)
```

你可以先粗暴记成：

```text
对 row-major compact layout，left_inverse 看起来像把 m 和 n 互换。
```

但更准确的原因是：

```text
offset 反解时，先解出 n，再解出 m。
```

### 2.8 什么时候不会这样

如果原 layout 本来就是 CuTe 的 coord-id 顺序：

```text
L = (2,3):(1,2)
```

也就是：

```text
L(m,n) = m + 2*n
```

那它已经等于：

```text
coord-id(m,n)
```

所以 `left_inverse(L)` 就更像 identity：

```text
offset -> offset
```

CuTe 可能把它化简成：

```text
6:1
```

或者其他等价的 identity-like layout。

所以不要把 `left_inverse` 永远理解成交换维度。

正确理解是：

```text
left_inverse(L):
    从 L 的 offset 反查回原坐标的 coord-id。
```

只是对：

```text
(M,N):(N,1)
```

这种 compact row-major layout，它刚好长得像：

```text
(N,M):(M,1)
```

### 2.9 从 layout 推到 inverse，再用 inverse 反推

这一节只做一件事：

```text
给一个 layout
-> 手算出它的 inverse
-> 再拿 inverse 验证能反推回来
```

还是用：

```text
L = (2,3):(3,1)
```

#### 第一步：正向 layout

`L` 的坐标是：

```text
(m,n)
```

shape 是：

```text
(2,3)
```

stride 是：

```text
(3,1)
```

所以：

```text
offset = 3*m + n
```

正向表：

```text
原坐标    L 算出来的 offset
(0,0)     0
(0,1)     1
(0,2)     2
(1,0)     3
(1,1)     4
(1,2)     5
```

#### 第二步：原坐标的 coord-id

同一个 shape `(2,3)` 的 coord-id 是：

```text
coord-id = m + 2*n
```

所以：

```text
原坐标    coord-id
(0,0)     0
(1,0)     1
(0,1)     2
(1,1)     3
(0,2)     4
(1,2)     5
```

把第一张表和第二张表合起来：

```text
原坐标    offset   coord-id
(0,0)     0        0
(0,1)     1        2
(0,2)     2        4
(1,0)     3        1
(1,1)     4        3
(1,2)     5        5
```

所以 inverse 必须满足：

```text
inverse(offset) = coord-id
```

也就是：

```text
offset    inverse 输出
0         0
1         2
2         4
3         1
4         3
5         5
```

写成一行：

```text
offset:   0  1  2  3  4  5
inverse:  0  2  4  1  3  5
```

#### 第三步：把 inverse 表达成 layout

我们要找一个 layout，输入 offset，输出上面这一行：

```text
0 2 4 1 3 5
```

原来的 offset 公式是：

```text
offset = 3*m + n
```

反解：

```text
n = offset % 3
m = offset / 3
```

所以 inverse 的输入坐标顺序是：

```text
(n,m)
```

`n` 有 3 种，`m` 有 2 种，所以 inverse 的 shape 是：

```text
(3,2)
```

inverse 的输出要是：

```text
coord-id = m + 2*n
```

但是 inverse 的输入坐标是 `(n,m)`，所以重写成：

```text
coord-id = 2*n + m
```

因此 inverse layout 是：

```text
Inv = (3,2):(2,1)
```

#### 第四步：用 inverse 反推验证

现在验证：

```text
Inv(L(m,n)) = coord-id(m,n)
```

先看原坐标 `(0,2)`。

正向：

```text
L(0,2) = 3*0 + 2 = 2
```

所以 offset 是：

```text
2
```

现在把 offset `2` 喂给：

```text
Inv = (3,2):(2,1)
```

注意：`Inv` 的 shape 是 `(3,2)`。

所以 scalar `2` 先按 shape `(3,2)` 解成坐标：

```text
2 -> (2,0)
```

因为：

```text
2 = 2 + 3*0
```

然后用 Inv 的 stride `(2,1)` 算：

```text
Inv(2,0) = 2*2 + 1*0 = 4
```

这个 `4` 是什么？

它就是原坐标 `(0,2)` 的 coord-id：

```text
coord-id(0,2) = 0 + 2*2 = 4
```

所以：

```text
Inv(L(0,2)) = 4
```

和：

```text
coord-id(0,2) = 4
```

对上。

再看原坐标 `(1,0)`。

正向：

```text
L(1,0) = 3*1 + 0 = 3
```

把 offset `3` 喂给 Inv。

`Inv` 的 shape 是 `(3,2)`，所以：

```text
3 -> (0,1)
```

因为：

```text
3 = 0 + 3*1
```

然后：

```text
Inv(0,1) = 2*0 + 1*1 = 1
```

而：

```text
coord-id(1,0) = 1 + 2*0 = 1
```

也对上。

完整验证表：

```text
原坐标   L 正推 offset   offset 按 Inv.shape 解码   Inv 输出   原 coord-id
(0,0)    0              0 -> (0,0)                0         0
(0,1)    1              1 -> (1,0)                2         2
(0,2)    2              2 -> (2,0)                4         4
(1,0)    3              3 -> (0,1)                1         1
(1,1)    4              4 -> (1,1)                3         3
(1,2)    5              5 -> (2,1)                5         5
```

所以这整个过程可以记成：

```text
L:
    原坐标 -> offset

left_inverse(L):
    offset -> 原坐标的 coord-id

Inv(L(原坐标)):
    原坐标 -> offset -> 原坐标的 coord-id
```

这就是源码注释里的：

```text
composition(left_inverse(L), L) == identity_layout(shape(L))
```

这里的 identity 不是返回 tuple `(m,n)`，而是返回这个坐标的 `coord-id`。

## 3. 一个 Python DSL 的坑

在当前 `nvidia-cutlass-dsl` / `flashinfer` 环境里：

```python
print_latex(left_inverse_layout)
```

对某些 `left_inverse` 生成的 rank-1 layout，比如：

```text
16:1
128:1
```

可能直接段错误。

这个更像是 Python DSL 的 `print_latex` 或 MLIR/FFI runtime bug，而不是 layout algebra 算错。

更稳的检查方式是：

```python
print(acc_layout_inv)
```

如果一定要画图，可以先把它 compose 到一个显式的 2D layout 上，或者只画后续 `composition` 的 rank-2 可视化结果。

## 4. composition / compose

`composition` 就是函数复合：

```text
C = composition(A, B)
C(x) = A(B(x))
```

也就是说：

```text
B: C 的坐标空间 -> A 能理解的坐标/coord-id
A: A 的坐标空间 -> 最终 offset
C: C 的坐标空间 -> 最终 offset
```

所以：

```text
B 决定结果 layout 的输入坐标系统
A 决定最终访问哪个 offset
```

一个比较一般的直觉是：

```text
composition(A, B) = 用 B 重新参数化 A
```

或者说：

```text
把 A 放到 B 这个坐标系统下观察。
```

它不移动数据，只构造一个新的 layout：

```text
新坐标 -> 原始数据 offset
```

## 5. scalar 输入多维 layout 时发生了什么

`composition(A, B)` 里最容易绕的地方是：

```text
B(x)
```

可能返回一个 scalar，而 `A` 是多维 layout。

这时 CuTe 会把这个 scalar 当成 `A` 的 coordinate-id 来解释，然后再用 `A` 的 stride 算最终 offset。

例如：

```text
A = (4,4):(4,1)
B = (8,2):(2,1)
```

对于坐标：

```text
x = (4,1)
```

先算：

```text
B(4,1) = 4 * 2 + 1 = 9
```

然后把 `9` 当成 `A.shape = (4,4)` 的 coordinate-id。

在我们讨论的这个例子里：

```text
9 -> (9 % 4, 9 / 4) = (1,2)
```
也就是将A当作col major的，如果A已经是col major了，则保持不变

然后用 `A` 的 stride 算：

```text
A(1,2) = 1 * 4 + 2 = 6
```

所以：

```text
composition(A, B)(4,1) = 6
```

这就是为什么图里会出现：

```text
9 -> (1,2) -> 6
```

注意：如果图里写成 `% N`，在 `M == N` 时看不出问题；更一般地说，要看它是在按哪个维度的 coordinate-id 规则解释。

## 6. 为什么 FlashAttention 风格代码用 inverse + compose

代码形如：

```cpp
auto acc_layout_inv = left_inverse(trC_as_C_layout);
auto a_layout_algebra = acc_layout_inv.compose(trC_as_A_layout);
```

等价于：

```text
a_layout_algebra(x) = acc_layout_inv(trC_as_A_layout(x))
```

可以这样理解：

```text
trC_as_A_layout:
    A fragment 坐标 -> accumulator-like linear id

acc_layout_inv:
    accumulator linear id -> accumulator coordinate-id

a_layout_algebra:
    A fragment 坐标 -> accumulator coordinate-id
```

所以它不是在搬数据，也不是生成数据。

它是在构造一个坐标翻译：

```text
A fragment 里的这个位置，对应 accumulator C layout 坐标系统里的哪个位置？
```

因此：

```cpp
left_inverse(C_layout).compose(A_layout)
```

的作用是：

```text
用 A fragment 的坐标系统，去表达 C accumulator 的坐标系统。
```

## 7. 关于 tCrA / tCrC 这类 layout

像：

```text
tCrA.shape  = ((_2,_2,_2), _4, _2)
tCrA.stride = ((_1,_2,_4), _8, _32)
```

不要直接把第一个 nested mode：

```text
(_2,_2,_2)
```

理解成 A 矩阵的普通二维 row/col。

它更像是 MMA atom 内部的 fragment value 编号拆分。

也就是说：

```text
((_2,_2,_2), _4, _2)
  ^^^^^^^^^   ^   ^
  atom 内部   MMA_M MMA_K
  value mode
```

其中：

```text
((_1,_2,_4), _8, _32)
```

说明 atom 内部 value mode 是连续编号的：

```text
0,1,2,3,4,5,6,7
```

外面的 `_4`、`_2` 才是在 MMA_M / MMA_K 方向上的重复。

所以这类 layout 描述的是：

```text
每个线程持有的 MMA register fragment 如何编号
```

而不是 shared memory 里 A tile 的普通 row-major/col-major 排布。

如果想看 fragment value 对应原始矩阵里的哪个 `(m,k)`，应该用 identity tensor 做 partition，例如：

```cpp
auto cA = make_identity_tensor(make_shape(Int<32>{}, Int<16>{}));
auto tCcA = thr_mma.partition_A(cA);
```

这样打印出来的才是 fragment value 到原始坐标的对应关系。

## 8. 最终心智模型

可以这样记：

```text
left_inverse(L):
    把 L 的 offset 映射反过来，回到 coordinate-id 空间。
    对 compact layout，经常看起来像 reshape / shape 重新解释。

composition(A, B):
    构造 C(x) = A(B(x))。
    B 给结果 layout 提供输入坐标系统。
    A 给最终 offset。
    不搬数据，只生成新的索引映射。
```
