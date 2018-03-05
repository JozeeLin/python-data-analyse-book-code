# NumPy高级应用

[TOC]

## ndarray对象的内部机理

> NumPy的ndarray提供一种将同质数据快解释为多维数组对象的方式。数据类型决定了数据的解释方式。narray内部由以下内容组成:
>
> - 一个指向数组(一个系统内存块)的指针
> - 数据类型或dtype
> - 一个表示数组形状(shape)的元组
> - 一个跨度元组(stride),其中的整数指的是为了前进到当前维度下一个元素需要'跨过'的字节数。例如，一个典型的(C顺序，稍后将详细讲解)3x4x5的float64(8个字节)数组，其跨度为(160,40,8)

```python
#数组形状
np.ones((10,5)).shape
#数组跨度
np.ones((3,4,5), dtype=np.float64).strides
```

### NumPy数据类型体系

```python
ints = np.ones(10, dtype=np.uint16)
floats = np.ones(10, dtype=np.float32)
#判断某个子类型是否属于某个超类的，使用issubdtype
np.issubdtype(ints.dtype, np.integer)
np.issubdtype(floats.dtype, np.floating)
#调用dtype的mro方法即可查看其所有的父类
np.float64.mro()
```



## 高级数组操作

> 花式索引、切片、布尔条件取子集等操作之外

### 数组重塑

> 无需复制任何数据，数组就能从一个形状转换为另一个形状。

```python
arr = np.arange(8)
#一维数组重塑
arr.reshape((4,2))
#多维数组重塑
arr.reshape((4,2)).reshape((2,4))
#作为参数的形状的其中一维可以是-1,它表示该维度的大小由数据本身推断而来，不需要自己算
arr = np.arange(15)
arr.reshape((5,-1))#等效于arr.reshape((5,3))
#数组的shape属性是一个元组
other_arr = np.ones((3,5))
other_arr.shape
arr.reshape(other_arr.shape)
#与reshape将一维数组转换为多维数组的运算过程相反的运算通常称为扁平化(flattening)或散开(raveling)
arr = np.arange(15).reshape((5,3))
arr.ravel() #ravel不会产生源数据的副本.
arr.flatten()#flatten方法的行为类似于ravel,但总是返回源数据的副本
```

### C和Fortran顺序

> 默认情况下，NumPy数组是按行优先顺序创建的。在空间方面，这就意味着，对于一个二维数组，每行中的数据项被存放在相邻内存位置上的。另一种顺序是列优先顺序，意味着每列中的数据项是被存放在相邻内存位置上的。行和列优先顺序又分别称为C和Fortran顺序。

```python
arr  = np.arange(12).reshape((3,4))
arr.ravel() #默认行优先，相当于arr.ravel('C')
arr.ravel('F') #列优先
```

### 数组的合并和拆分

> numpy.concatenate按指定轴将一个由数组组成的序列连接到一起。

```python
arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[7,8,9],[10,11,12]])
np.concatenate([arr1,arr2], axis=0)#拼接成行数据，两个数组的形状的行大小要想等
np.vstack((arr1,arr2))#等效于上面那一句
np.concatenate([arr1,arr2], axis=1)#拼接成列数据，两个数组的形状的列大小要想等
np.hstack((arr1,arr2))#等效于上面那一句

arr = randn(5,2)
#split用于将一个数组沿指定轴拆分为多个数组
first,second,third = np.split(arr,[1,3])
print first
print 
print second
print
print third
```

**数组连接函数:**

- concatenate——最一般化的连接，沿一条轴连接一组数组
- vstack、row_stack——以面向行的方式对数组进行堆叠(沿轴0)
- hstack——以面向列的方式对数组进行堆叠(沿轴1)
- column_stack——类似于hstack，但是会先将一维数组转换为二维列向量
- dstack——以面向"深度"的方式对数组进行堆叠(沿轴2)
- split——沿指定轴在指定的位置拆分数组
- hsplit、vsplit、dsplit——split的便携化函数，分别沿轴0、轴1、轴2进行拆分

```python
#堆叠辅助类：r_和c_ #类似于前面的concatenate操作
arr = np.arange(6)
arr1 = arr.reshape((3,2))
arr2 = randn(3,2)
np.r_[arr1,arr2] #r_相当于vstack
np.c_[np.r_[arr1,arr2],arr] #c_相当于hstack
#还可以将切片翻译为数组
np.c_[1:6,-10:-5]
```

### 元素的重复操作:tile和repeat

```python
#repeat会将数组中的各个元素重复一定次数，从而产生一个更大的数组
arr = np.arange(3)
#各元素重复相同的次数
arr.repeat(3)
#各元素重复不同的次数
arr.repeat([2,3,4])
#对于多维数组，还可以让它们的元素沿指定轴重复
arr = randn(2,2)
arr.repeat(2，axis=0)
#没有设置轴向，则数组会被扁平化
arr.repeat(2)
#对多维进行重复
arr.repeat([2,3], axis=0) #沿着行重复
arr.repeat([2,3], axis=1)#沿着列重复
#tile的功能是沿指定轴堆叠数组的副本
np.tile(arr,2) #类似于np.hstack((arr,arr)),但是tile更加方便，只需要指定重复次数，而hstack需要把arr写重复次数次
#第二个参数是瓷砖的数量
np.tile(arr,(2,1)) #重复2行1列
np.tile(arr,(3,2)) #重复3行2列
```

### 花式索引的等价函数:take和put

> 获取和设置数组子集的一个办法是通过整数数组使用花式索引

```python
arr = np.arange(10)*100
inds = [7,1,2,6]
arr[inds]
#ndarray有两个方法专门用于获取和设置单个轴向上的选区
arr.take(inds) #获取
arr.put(inds, 42) #设置
arr.put(inds, [40,41,42,43])
#在其他轴上使用take，需传入axis关键字
inds = [2,0,2,1]
arr = randn(2,4)
arr.take(inds, axis=1)
#put不接受axis参数，它只会在数组的扁平化版本上进行索引
```

## 广播

> 广播指的是不同形状的数组之间的算术运算的执行方式。
>
> 广播的原则：如果两个数组的后缘维度(trailing dimension, 即从末尾开始算起的维度)的轴长度相符或其中一方的长度为1,则认为它们是广播兼容的。广播会在缺失和(或)长度为1的维度上进行。

```python
arr = np.arange(5)
#乘法运算中，标量值4被广播到了其他所有的元素上
arr*4
#通过减去列平均值的方式对数组的每一列进行距平化处理
arr =randn(4,3)
arr.mean(0).shape #(3,)
demeaned = arr - arr.mean(0) #参数0表示求每一列的均值
demeaned.mean(0) #都为0

#对各行减去均值，进行距平化处理
row_means = arr.mean(1)
row_means.shape #(4,)
row_means.reshape((4,1))
demeaned = arr - row_means.reshape((4,1))
demeaned.mean(1) #都为0

#通过特殊的np.newaxis属性以及'全'切片来插入新轴
arr = np.zeros((4,4))
arr_3d = arr[:, np.newaxis, :] #三维数组，插入新轴
arr_3d.shape
#一维数组通过插入新轴创建二维数组
arr_1d = np.random.normal(size=3)
arr_1d[:,np.newaxis]
arr_1d[np.newaxis, :]
#对三维数组的轴2进行距平化
arr = randn(3,4,5)
depth_means = arr.mean(2)
demeaned = arr-depth_means[:, :, np.newaxis]

def demean_axis(arr, axis=0):
    means = arr.mean(axis)
    indexer = [slice(None)] * arr.ndim
    indexer[axis] = np.newaxis
    return arr-means[indexer]
```

### 通过广播设置数组的值

> 算术运算所遵循的广播原则同样也适用于通过索引机制设置数组值的操作。

```python
arr = np.zeros((4,3))
arr[:] = 5
col = np.array([1.28, -0.42, 0.44, 1.6])
arr[:] = col[:, np.newaxis]
arr[:2] = [[-1.37],[0.509]]
```

## ufunc高级应用

```python
#对数组中各个元素进行求和
arr = np.arange(10)
np.add.reduce(arr)
arr.sum()

arr = randn(5,5)
#对部分行进行排序
arr[::2].sort(1)
arr[:,:-1] < arr[:,1:]
np.logical_and.reduce(arr[:, :-1]<arr[:,1:], axis=1)
#accumulate跟reduce的关系就像cumsum跟sum的关系那样。它产生一个跟原数组大小相同的中间"累计"值数组
arr = np.arange(15).reshape((3,5))
np.add.accumulate(arr, axis=1)
#outer用于计算两个数组的叉积
arr=np.arange(3).repeat([1,2,2])
np.multiply.outer(arr, np.arange(5))
#outer输出结果的维度是两个输入数据的维度之和
result = np.subtract.outer(randn(3,4), randn(5))
result.shape
#reduceat用于计算"局部约简"，其实就是一个对数据各切片进行聚合的groupby运算。
arr = np.arange(10)
np.add.reduceat(arr, [0,5,8]) #根据切片范围求和

arr = np.multiply.outer(np.arange(4), np.arange(5))
np.add.reduceat(arr, [0,2,4], axis=1)
```

**ufunc的方法:**

- reduce(x)——通过连续执行原始运算的方式对值进行聚合
- accumulate(x)——聚合值，保留所有局部聚合结果
- reduceat(x, bins)——"局部"约简(也就是groupby)。约简数据的各个切片以产生聚合型数组
- outer(x,y)——对x和y中的每对元素应用原始运算。结果数组的形状为x.shape+y.shape

### 自定义ufunc

```python
def add_elements(x,y):
    return x+y
#用frompyfunc创建的函数总是返回Python对象数组
add_them = np.frompyfunc(add_elements, 2,1)
add_them(np.arange(8), np.arange(8))

add_them = np.vectorize(add_elements, otypes=[np.float64])
add_them(np.arange(8), np.arange(8))

arr = randn(10000)
%timeit add_them(arr, arr)
%timeit np.add(arr, arr)
```

## 结构化和记录式数组

> ndarray都是一种同质数据容器，也就是说，在它所表示的内存块中，各元素占用的字节数相同。
>
> 结构化数组是一种特殊的ndarray，其中各个元素可以被看做C语言中的结构体或SQL表中带有多个命名字段的行

```python
dtype = [('x', np.float64), ('y', np.int32)]
sarr = np.array([(1.5, 6), (np.pi, -2)], dtype=dtype)
#定义结构化dtype的方式最典型的办法是元组列表，各元组的格式为(field_anem, field_data_type)

#结构化对象中各个元素可以像字典那样进行访问，字段名保存在dtype.names属性中。
sarr[0]
sarr[0]['y']
sarr['x']
```

### 嵌套dtype和多维字段

```python
dtype = [('x',np.int64, 3), ('y', np.int32)]
arr = np.zeros(3, dtype=dtype)
#各个记录的x字段所表示的是一个长度为3的数组
arr[0]['x']
#访问arr['x']即可得到一个二维数组
data['x']

#嵌套dtype
dtype = [('x',[('a','f8'),('b','f4')]), ('y',np.int32)]
data = np.array([((1,2), 5), ((3,4),6)], dtype=dtype)
data['x']
data['y']
data['x']['a']
```

## 更多有关排序的话题

```python
arr = randn(6)
#就地排序
arr.sort()
arr = randn(3,5)
arr[:, 0].sort()
#np.sort会创建一个排好序的副本
arr = randn(5)
np.sort(arr)
#指定轴向对各块数据进行单独排序
arr = randn(3,5)
arr.sort(axis=1)
#返回反序的narray
arr[:, ::-1] #同行的反序
```

### 间接排序:argsort和lexsort

```python
values = np.array([5,0,1,3,2])
#根据values顺序对index的重排
indexer = values.argsort()
values[indexer]

arr = randn(3,5)
arr[0] = values
arr[:, arr[0].argsort()]
#lexsort可以一次性对多个键数组执行间接排序(字典序)
first_name = np.array(['Bob','Jane','Steve','Bill','Barbara'])
last_name = np.array(['Jones','Arnold','Arnold','Jones','Walters'])
sorter = np.lexsort((first_name, last_name)) #对last_name数组进行间接排序，并返回重排后的索引
zip(last_name[sorter], first_name[sorter])
```

### 其他排序算法

```python
values = np.array(['2:first','2:second','1:first','1:second','1:third'])
key = np.array([2,2,1,1,1])
indexer = key.argsort(kind='mergesort')
values.take(indexer)
#searchsorted是一个在有序数组上执行二分查找的数组方法，只要将值插入到它返回的那个位置就能维持数组的有序性
arr = np.array([0,1,7,12,15])
arr.searchsorted(9)
#传入一组值就能得到一组索引
arr.searchsorted([0,8,11,16])
#对于元素0,serachsorted会返回0.其默认行为是返回相等值组的左侧索引
arr = np.array([0,0,0,1,1,1,1])
arr.searchsorted([0,1])
arr.searchsorted([0,1], side='right')
#使用表示'面元边界'的数组将数据数组拆分开
data = np.floor(np.random.uniform(0,10000,size=50))
bins = np.array([0,100,1000,5000,10000])
#使用searchsorted得到各数据点所属区间的编号
labels = bins.searchsorted(data)
labels = np.digitize(data, bins) #使用numpy。digitize也可用于计算面元编号
#使用groupby可以对原数据集进行拆分
Series(data).groupby(labels).mean()
```

## NumPy的matrix类

## 高级数组输入输出

> 内存映像(memory map)用于处理在内存中放不下的数据集。

### 内存映像文件

> 内存映像文件是一种将磁盘上的非常大的二进制数据文件当做内存中的数组进行处理的方式。memmap对象允许将大文件分成小段进行读写，而不是一次性将整个数组读入内存。

```python
mmap = np.memmap('mymmp', dtype='float64', mode='w+', shape=(10000,10000))
section = mmap[:5]
section[:] = np.random.randn(5,10000)
mmap.flush() #写入磁盘
del mmap
#打开一个已经存在的内存映射时，仍然需要指明数据类型和形状
mmap = np.memmap('mymmp', dtype='float64', shape=(10000,10000))
```

### HDF5及其他数组存储方式

> PyTables和h5py这两个python项目可以将numpy的数组数据存储为高效且可压缩的HDF5格式(HDF意思是‘层次化数据格式’)
>
> PtTables提供了一些用于结构化数组的高级查询功能，而且还能添加列索引一提升查询速度。这跟关系型数据库提供的表索引功能非常相似。

## 性能建议

> - 将python循环和条件逻辑转换为数组运算和布尔数组运算
> - 尽量使用广播
> - 避免复制数据，尽量使用数组视图(即切片)
> - 利用ufunc及其各种方法

Cython、f2py、C