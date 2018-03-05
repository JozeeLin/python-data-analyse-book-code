# Numpy基础：数组和矢量计算

[TOC]

## Numpy的narray：一种多维数组对象

> 可以利用这样的数组对整块数据进行数学运算，其语法跟标量元素之间的运算一样

```python
data = np.array([[0.888,123,121],[2,3,4]])
data*10
```

### narray各种属性的获取方法

``` python
#获取数据形状
data.shape
#获取行数
data.ndim
#获取每列数据的类型
data.dtype
```

### narray的创建方法

``` python
#生成内容都是0的narray
np.zeros(10)
np.zeros((3,6))
#生成内容都是一些未初始化的随机数
np.empty((2,3,2))
#生成指定范围的所有实数,类似于朋友python的range函数
np.arange(15)
```

- **array** 输入数据为列表、元组、数组、或其他序列类型转换为ndarray，可以指定元素类型。默认直接复制输入数据
- **asarray** 将输入数据转为narray，如果输入数据本身就是narray类型，就不进行复制
- **arange** 类似于内置的range，返回的是列表
- **ones** 、**ones_like** 根据指定的形状和dtype返回一个全1数组。ones_like以另一个数组为参数，并根据其形状和dtype生成全1数组
- **zeros**、**zeros_like** 同ones一样，但是返回的是一个全0数组
- **empty**、**empty_like** 创建新数组，只分配内存空间但不填充任何值
- **eye** 、**identity** 创建一个N×N的方阵，对角线为1,其余为0

### narray类型转换——astype

``` python
#将string_ 类型转成32位整型，这也仅限于字符串表示的全是数字
arr = np.array(['1','2','3'])
arr.astype(np.int32)
```

- 转换出错类型TypeError
- 结果是创建出新的数组

### 数组和标量之间的运算

> 数组很重要，它使你不用编写循环即可对数据执行批量运算。这通常叫做矢量运算。大小相等(shape一样)的数组之间的任何算术运算都会将运算应用到元素级

```python
arr = np.array[[1,2,3],[4,5,6]]
arr*arr
'''array([[ 1,  4,  9],
       [16, 25, 36]])'''
arr-arr
'''array([[0, 0, 0],
       [0, 0, 0]])'''
1/arr
#之所以其余都为0,因为元素类型是整数
'''array([[1, 0, 0],
       [0, 0, 0]])'''
```

### 基本的索引和切片

#### 切片性质

- 对切片赋值，就是对切片里每一个元素都赋予同一个值

- 数组的切片是原始数组的视图。这意味着数据不会被复制，视图上的任何修改都会直接反映到源数组上。

- 元素访问(返回值都是原始数组的视图)

  ```python
  arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
  #读取第三行
  arr2d[2]
  #读取第一行第三列
  arr2d[0][2]
  arr2d[0,2]
  ```

- 多维数组的切片

  ```python
  arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
  #取第三行
  arr2d[2]
  '''array([7, 8, 9])'''
  #从第二行开始取，第三行前结束
  arr2d[1:2]
  '''array([[4, 5, 6]])'''
  #从第一行开始取到第二行止，取从第一列还是到结束
  arr2d[:2,1:]
  '''array([[2, 3],
         [5, 6]])'''
  ```

- 布尔索引(布尔索引可以跟整数索引共用)

  > numpy.random.randn函数产生一些正态分布的随机数据

  - 通过布尔值来进行切片

    ``` python
    names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
    names == 'Bob'
    '''
    array([ True, False, False,  True, False, False, False], dtype=bool)
    '''
    data = np.random.randn(7,4)
    #表示取数组元素值为True的行，根据上图显示输出的应该是第1、4行
    #注意：上面的布尔数组长度必须跟data数组的行数相同
    data[names == 'Bob']
    ```

  - 通过布尔值数组设置值

    ```python
    #对数组中小于0的元素赋值为0
    data[data<0] = 0
    #使用布尔数组进行切片，筛选出行集合，然后对被选中的所有行进行赋值
    data[names != 'Joe'] = 7
    ```

- 花式索引(利用整数数组进行索引)，但是返回值是新创建的数组，不是原始数组的视图

  ```python
  arr = np.empty((8,4))
  for i in range(8):
      arr[i] = i
  arr[[4,3,0,6]]
  '''
  array([[ 4.,  4.,  4.,  4.],
         [ 3.,  3.,  3.,  3.],
         [ 0.,  0.,  0.,  0.],
         [ 6.,  6.,  6.,  6.]])
  '''
  #使用负数就会从末尾开始获取行
  arr[[-3,-5,-7]]
  '''
  array([[ 5.,  5.,  5.,  5.],
         [ 3.,  3.,  3.,  3.],
         [ 1.,  1.,  1.,  1.]])
  '''
  #获取指定元素，(1,0)、(5,3)、(7,1)、(2,2)
  arr[[1,5,7,2],[0,3,1,2]]
  #获取指定区域所有的元素,1,5,7,2行0,3,1,2列
  '''array([[ 4,  7,  5,  6],
         [20, 23, 21, 22],
         [28, 31, 29, 30],
         [ 8, 11,  9, 10]])'''
  ```

- 数组转置和轴对换(转置返回源数据视图)

  ```python
  arr = np.arange(15).reshape((3,5))
  #数组的转置，矩阵运算
  arr.T
  #矩阵的内积运算
  np.dot(arr.T, arr)
  #transpose就是对轴标号进行重新排列
  arr = np.arange(24).reshape((3,2,4))
  #由原数组变成格式为(4,3,2)形状的数组
  arr.transpose((2,0,1))
  #表示变成形状为(4,2,3)形状的数组
  arr.transpose()
  #跟transpose类似，但是只是用来对两个轴进行转换.比如，对1,2进行转换，变成(3,4,2)
  arr.swapaxes(1,2)
  ```

## 通用函数：快速的元素级数组函数

> 通用函数是对narray数组数据执行元素级运算的函数。

1. 一元通用函数

   - np.sqrt 求每个元素的开方
   - abs(fabs速度更快)计算整数、浮点书或复数的绝对值
   - sqrt 平方跟 $\sqrt{x}$
   - square 平方 $x^2$
   - exp  平方 $e^x$
   - log,log10,log2,log1p 分别为$\log_e{x}$、$\log_{10}{x}$、$\log_2{x}$、$\log_{1+x}{x}$
   - sign 计算各元素的正负号:1(正数)，0(零)，-1(负数)
   - ceil 计算各元素的ceiling值，即大于等于该值的最小值(向上取整)
   - floor 计算各元素的floor值，即小于等于该值的最大值(向下取整)
   - rint 将元素四舍五入到最接近的整数、dtype不变
   - modf 将数组的小数和整数部分以两个独立数组的形式返回
   - isnan 返回一个表示“哪些值是NaN”的布尔数组
   - isfinite、isinf 分别返回表示"哪些元素是有穷的(非inf，非NaN)"，"哪些元素是无穷的"的布尔型数组
   - cos、cosh、sin、sinh、tan、tanh 普通型和双曲型函数
   - arccos、arccosh、arcsin、arcsinh、arctan、arctanh 反函数
   - logical_not 计算各元素not x的真值。相当与-arr，求各元素的逻辑非

2. 二元通用函数

   - add 将两个数组之间一一对应的元素相加

   - subtract 将两个数组之间一一对应的元素相减

   - multiply 数组元素相乘

   - divide、floor_divide 除法或向下圆整除法(丢弃余数)

   - power 对第一个数组中的元素A， 根据第二个数组中的相应元素B，计算$A^B$

   - maximum、fmax 求两个数组中的最大值数组，fmax忽略NaN。要求返回值的数组形状和参数数组形状一样

   - minimum、fmin 同上，求最小值

   - mod 元素级的求模计算

   - copysign 将第二个数组中的值的符号复制给第一个数组中的值

   - greater、greater_equal、less、less_equal、equal、not_equal 比较运算，相当于$> , >=,<,<=,==,!=$

   - logical_and, logical_or, logical_xor 逻辑运算，相当于$\&$,$|$,^


## 利用数组进行数据处理

> numpy 数组可以使你将许多种数据处理任务表述为简洁的数组表达式，否则需要编写循环。使用数组表达式来替代循环的做法叫做矢量化。

```python
points1 = np.arange(-5,5,1)
points2 = np.arange(0,4,1)
xs, ys = np.meshgrid(points1, points2)
xs
'''array([[-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4]])'''
ys
'''array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])'''
#计算sqrt(x^2+y^2)
z = np.sqrt(xs**2+ys**2)
```

### 将条件逻辑表述为数组运算

> numpy.where函数是三元表达式 x if condition else y的矢量化版本。例如：

```python
xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr = np.array([2.1,2.2,2.3,2.4,2.5])
cond = np.array([True, False, True, True, False])
result = np.where(cond, xarr, yarr)
result
'''
array([ 1.1,  2.2,  1.3,  1.4,  2.5])
'''
```

### 数学和统计方法(以下函数会强制把布尔型转成整数)

> 通过数组上的一组数学函数对整个数组或某个轴向的数据进行统计计算。

- sum 对数组中全部或某轴向的元素求和。零长度的数组的sum为0
- mean 算术平均数。零长度的数组的mean为NaN
- std、var 分别为标准差和方差，自由度可调(默认为n)
- min、max 最大值和最小值
- argmin、argmax 分别为最大和最小元素的索引
- cumsum 所有元素的累计和
- cumprob 所有元素的累计积

### 用于布尔型数组的方法(也可应用于非布尔情况，非0元素都当作True)

- any用于测试数组中是否存在一个或多个True
- all 检查数组中所有值是否都是True

### 排序

- sort 可以对数组的任何轴进行排序，参数为轴编号，返回值是排好序的副本

### 唯一化以及其他的集合逻辑

- unique 用于找到数组中的唯一值并返回已排序的结果

- in1d 用于测试一个数组中的值在另一个数组中的成员资格，返回值是布尔型数组

  ```python
  values = np.array([6,0,0,3,2,5,6])
  np.in1d(values, [2,3,6])
  '''array([ True, False, False,  True,  True, False,  True], dtype=bool)
  '''
  ```

- intersect1d(x,y) 计算x和y中的公共元素，并返回有序结果

- union1d(x,y) 计算x和y的并集，并返回有序结果

- setdiff1d(x,y) 集合的差，即元素在x中且不在y中

  ```python
  values = np.array([6,0,0,3,2,5,6])
  np.setdiff1d(values,[2,3,6])
  '''
  array([0, 5])
  '''
  ```

  ​

- setxor1d(x,y) 集合的对称差，即存在于一个数组中但不同时存在于两个数组中的元素

## 用于数组的文件输入输出

### 将数组以二进制格式保存到磁盘

- np.save、np.load是读写磁盘数组数据的两个主要函数。默认情况下，数组以未压缩的原始二进制格式保存在扩展名为.npy的文件中。

### 存取文本文件

- np.loadtxt(filename, delimiter=分隔符)
- np.savetxt执行相反的操作，保存txt文件
- np.genfromtxt跟loadtxt差不多，但是它面向结构化数组和缺失数据处理

## 线性代数

- np.dot 内积函数
- diag 以一维数组的形式返回方阵的对角线(或非对角线)元素，或将一维数组转换为方阵(非对角线元素为0)
- trace 计算对角线元素的和
- det 计算矩阵行列式
- eig 计算方阵的本征值和本征向量
- inv 计算矩阵的逆
- pinv 计算矩阵的Moore-Penrose伪逆
- qr 计算QR分解
- svd 计算奇异值分解(SVD)
- solve 解线性方程组Ax = b, 其中A为一个方阵
- lstsq  计算Ax=b的最小二乘解

## 随机数生成

> numpy.random模块对python内置的random进行补充，增加了一些用于高效生成多种概率分布的样本值的函数。

```python
#生成标准正太分布的4×4样本数组
samples = np.random.normal(size=(4,4))
```

- seed 确定随机数生成器的种子
- permutation 返回一个序列的随机排列或返回一个随机排列的范围
- shuffle 对一个序列就地随机排列
- rand 产生均匀分布的样本值
- randint 从给定的上下限范围内随机选取整数
- randn 产生正态分布(平均值为0, 标准差为1)的样本值
- binomial 产生二项分布的样本值
- normal 产生正态(高斯)分布的样本值
- beta 产生Beta分布的样本值
- chisquare 产生卡方分布的样本值
- gamma 产生gamma分布的样本值
- uniform 产生在[0,1)中均匀分布的样本值