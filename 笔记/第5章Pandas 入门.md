# pandas 入门

[TOC]

## 引言

构建pandas的目的：

- 具备按轴自动或显式数据对齐功能的数据结构。这可以防止许多由于数据未对齐以及来自不同数据源(索引方式不同)的数据而导致的常见错误
- 集成时间序列功能
- 既能处理时间序列数据也能处理非时间序列数据的数据结构
- 数学运算和约简可以根据不同的元数据(轴编号)执行
- 灵活处理缺失值
- 合并及其他出现在常见数据库中的关系型运算

## pandas的数据结构介绍

### Series

> Series是一种类似于一维数组的对象，它由一组数据(各种numpy数据类型)以及一组与之相关的数据标签(即索引)组成。

```python
#生成Series，索引为默认的整数索引，从0开始
obj = Series([4,7,-5,3])
'''
0    4
1    7
2   -5
3    3
dtype: int64
'''
#获取obj的数组表示形式
obj.values
'''
array([ 4,  7, -5,  3])
'''
#获取obj的索引对象
obj.index
'''
RangeIndex(start=0, stop=4, step=1)
'''
#指定索引
obj = Series([4,7,-5,3],index=['a','b','c','d'])
'''
a    4
b    7
c   -5
d    3
dtype: int64'''
#通过索引访问Series对象中的元素值
obj['a']
'''4'''
obj[['b','a','c']]
'''
b    7
a    4
c   -5
dtype: int64
'''
#对Series中的元素赋值
obj['b'] = 10
'''
a     4
b    10
c    -5
d     3
dtype: int64
'''
obj[['b','a','c']] = 7
'''
a    7
b    7
c    7
d    3
dtype: int64
'''
#判断索引a是否存在于Series obj中
'a' in obj
```

#### Series 性质

- numpy数组运算都会保持索引和值之间的链接
- Series看成是一个定长的有序字典，它可以用在许多原本需要字典参数的函数中
- 可以通过字典来直接创建Series
- 算术运算中会自动对齐不同的索引

### DataFrame

> DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型(数值、字符串、布尔值等)

#### DataFrame的构建

```python
data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],
       'year':[2000,2001,2002,2001,2002],
       'pop':[1.5,1.7,3.6,2.4,2.9]}
frame = DataFrame(data)
#列操作,返回值分别为列表，Series类型
frame.year
frame['state'] #返回的是源数据的视图
#列操作，返回值为DataFrame
frame[['state','year']]
#行操作，返回值为Series
frame.loc[0]
#行操作，返回值为DataFrame，注意，通过转置对行操作可以转变成列操作
frame.loc[[0,1,2]]
#对列赋值,注意：将列表、数组、Series赋值给列时，其长度必须跟列的长度一样，同时Series会进行索引对齐，为不存在的列赋值会创建新的列。
frame['pop'] = [1,2,2,3,4]#注意，赋值时只能使用字典标记的方式
#列删除
del frame['pop']
#查看当前DataFrame的所有列名称
frame.columns
#使用嵌套字典创建DataFrame,解释1：外层字典的键作为列，内层字典的键作为行索引。解释2：相较于前面的非嵌套字典，它是在之前的基础上，预设了行索引
pop = {'Nevada':{2001:2.4, 2002:2.9},
      'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
frame2 = DataFrame(pop)
#字典的值为Series格式的时候，会进行索引对齐
pdata = {'Ohio':frame2['Ohio'][:-1],'Nevada':frame2['Nevada'][:2]}
frame3 = DataFrame(pdata)
#设置index和columns的name属性
frame.index.name = 'year'
frame.columns.name = 'state'
#values属性,返回二维ndarray形式的源数据
frame.values
```

**更多的构造手段:**

- 二维ndarray——数据矩阵，还可以传入行标和列标
- 由数组、列表或元组组成的字典——每个序列会变成DataFrame的一列。所有序列的长度必须相同
- Numpy的结构化/记录数组——类似于“有数组组成的字典”
- 由Series组成的字典——每个Series会成为一列，如果没有显式的指定索引，则各Series的索引会被合并成结果的行索引
- 由字典组成的字典——各内层字典会成为一列。键会被合并成结果的行索引，同Series一样
- 字典或Series的列表——各项将会成为DataFrame的一行。字典键或Series索引的并集将会成为DataFrame的列标
- 由列表或元组组成的列表——类似于“二维ndarray”
- 另一个DataFrame——该DataFrame的索引将会被沿用，除非显式指定了索引
- Numpy的MaskedArray——类似于“二维ndarray”，只是掩码值在结果DataFrame会变成NA/缺失值

#### 索引对象

> DataFrame的索引对象负责管理轴标签和其他元数据(比如轴名称等)。构建DataFrame时，所用到的任何数组或其他序列的标签都会被转换成一个Index

**性质：**

- 不可修改，便于多个数据结构之间安全共享
- 可继承性，用于实现轴索引功能

**相关类：**

- Index——最泛化的Index对象，将轴标签表示为一个由Python对象组成的Numpy数组
- Int64Index——针对整数的特殊Index
- MultiIndex——‘层次化’索引对象，表示单个轴上的多层索引。可以看做由元组组成的数组
- DatetimeIndex——存储纳秒级时间戳(用Numpy的datetime64类型表示)
- PeriodIndex——针对Period数据(时间间隔)的特殊Index

**方法和属性:**

- append——连接另一个Index对象，产生一个新的Index
- diff——计算差集，并得到一个Index
- intersection——计算交集
- union——计算并集
- isin——计算一个指示各值是否都包含在参数集合中的布尔型数组
- delete——删除索引i处的元素，并得到新的Index
- drop——删除传入的值，并得到新的Index
- insert——将元素插入到索引i处，并得到新的Index
- is_monotonic——当各元素均大于等于前一个元素时，返回True
- is_unique——当Index没有重复值时，返回True
- unique——计算Index中唯一值的数组
- drop_duplicates——删除重复索引
- duplicated——判断那些索引是重复的，设置为False，返回布尔数组

## 基本功能

### 重新索引

- **reindex**，其作用是创建一个适应新索引的新对象

  **参数：**

  - index—用作索引的新序列。既可以是Index实例，也可以是其他序列型的Python数据结构。Index会被完全使用，就像没有任何复制一样
  - method 插值(填充)方式
  - fill_value 在重新索引的过程中，需要引入缺失值时使用的替代值
  - limit 前向或后向填充时的最大填充量
  - level 在MultiIndex的指定级别上匹配简单索引，否则选取其子集
  - copy 默认为True，无论如何都复制;如果为False，则新旧相等就不复制

```python
obj = Series([1,2,3,4], index=['a','b','c','d'])
#根据新索引进行重排,如果索引长度大于原索引的长度，多出来的部分用fill_value来填充，同样需要满足索引对齐原则
obj2 = obj.reindex(['a','b','c','d','e'], fill_value=0)
#时间序列重新索引需要做一些插值处理，method方法
obj3 = Series(['blue','purple','yellow'], index=[0,2,4])
obj3.reindex(range(6), method='ffill')
#重新设置列索引,也遵循对齐原则
frame = DataFrame(np.arange(9).reshape((3,3)), index=['a','c','d'], columns=['Ohio','Texas','California'])
states = ['Texas','Utah','California']
frame2 = frame.reindex(columns=states)
```

> reindex的(插值)method选项，只对行索引有作用
>
> 注意：在使用method方法是，必须保证需要修改的index或者columns都是有序的。所以传给index和columns的参数最好先排好序。
>
> 1、ffill或pad    前向填充(或搬运)值，使用NaN值前面最近的非NaN值来进行填充
>
> 2、bfill或backfill    后向填充(或搬运)值

### 丢弃指定轴上的项

> drop 方法返回的是一个新对象。参数是索引数组或者索引列表

```python
#Series
obj = Series(np.arange(5.), index=['a','b','c','d','e'])
new_obj = obj.drop('c')
#DataFrame
data = DataFrame(np.arange(16).reshape((4,4)), index=['Ohio','Colorado','Utah','New York'],
                columns=['one','two','three','four'])
#删除行
data.drop(['Colorado','Ohio'])
#删除列,axis默认为0
data.drop(['two'], axis=1)
```

### 索引、选取和过滤

- Series (Series索引类似与numpy数组，但是索引类型不局限于整数)

  ```python
  obj = Series(np.arange(4.), index=['a','b','c','d'])
  #使用索引名(标签)
  obj['a']
  #使用索引编号
  obj[0]
  #利用标签进行切片,它会包含末端，区别与编号切片
  obj['a':'c']
  '''
  a    0.0
  b    1.0
  c    2.0
  dtype: float64
  '''
  #利用编号切片
  obj[0:2]
  '''
  a    0.0
  b    1.0
  dtype: float64
  '''
  #设置值遵循广播性质
  obj['b':'c'] = 5
  obj
  '''
  a    0.0
  b    5.0
  c    5.0
  d    3.0
  dtype: float64
  '''
  ```

- DataFrame

  ```python
  data = DataFrame(np.arange(16).reshape((4,4)),
                  index=['Ohio','Colorado','Utah','New York'],
                  columns=['one','two','three','four'])
  #列索引,返回值为DF
  data[['one','two']]
  #行索引，返回值为DF
  data[:2] #切片
  data[data['two']>5] #布尔数组
  #DataFrame索引，返回DF
  data[data<5] #返回与原DF形状一样的数据，但是小于5的数都用NaN填充
  #标签索引，行列索引，类似于二维数组的读取方式，不能跟编号标签索引混用
  data.loc[['Ohio'],:] #表示选取Ohio行的所有列
  data.loc[:,['two']] #表示选取two列的所有行
  data.loc[['Ohio'],['two']] #选取Ohio行two列的值
  data.loc[:'Utah',:'two']  #使用标签切片来进行索引
  data.loc[data.three>5, :]  #使用布尔数组来进行索引
  #另一个标签索引方法,使用编号或者编号数组来进行索引
  data.iloc[:2,:2]  #可以用编号切片来进行索引
  ```

### 算术运算和数据对齐

> 不同索引的对象进行算术运算，在将对象相加时，如果存在不同的索引时，结果的索引就是该索引对的并集。

```python
#Series
s1 = Series([1,2,3,4], index=['a','c','b','d'])
s2 = Series([11,22,33,44,55], index = ['a','c','e','b','f'])
s1+s2  #注意NaN跟跟任何值进行运算结果都为NaN
'''
a    12.0
b    47.0
c    24.0
d     NaN
e     NaN
f     NaN
dtype: float64
'''
#DataFrame跟Series一样的原理
```

#### 在算术方法中填充值

```python
s1 = Series([1,2,3,4], index=['a','c','b','d'])
s2 = Series([11,22,33,44,55], index = ['a','c','e','b','f'])
#下面的效果等同于，先让两个对象的索引对齐，然后对原对象不存在的索引赋值为0
s1.add(s2, fill_value=0)
'''
a    12.0
b    47.0
c    24.0
d     4.0
e    33.0
f    55.0
dtype: float64
'''
#同样的在进行重新索引的时候，可以指定填充值
s1.reindex(s2.index, fill_value=0)
```

#### DataFrame和Series之间的运算

```python
frame = DataFrame(np.arange(12.).reshape((4,3)), columns=list('bde'),index=['Utah','Ohio','Texas','Oregon'])
#匹配列在行上广播(根据列索引对齐原则和广播性质,同样的不能对齐的索引结果为NaN)
series = frame.iloc[0]
frame - series
'''
		b	d	e
Utah	0.0	0.0	0.0
Ohio	3.0	3.0	3.0
Texas	6.0	6.0	6.0
Oregon	9.0	9.0	9.0
'''
#匹配行且在列上广播
series2 = frame['d'] #必须返回Series格式
frame.sub(series2, axis=0) #axis=0表示跟行索引对齐(在列上广播)，axis=1表示跟列索引对齐(在行上广播)
```

### 函数应用和映射

> NumPy的ufuns(元素级数组方法)也可用于操作pandas对象

```python
frame = DataFrame(np.random.randn(4,3), columns=list('bde'), index=['Utah','Ohio','Texas','Oregon'])
#求出DataFrame所有元素的绝对值
np.abs(frame)
#对行或列应用函数，使用apply来完成
f = lambda x: x.max()-x.min()  #这里X表示一行或一列的数组
#针对行来处理
frame.apply(f)
#针对列来处理
frame.apply(f, axis=1)
```

#### 排序和排名

```python
#按照行或列索引进行排序，axis=1列，默认行。ascending=False降序，默认升序
#Series
obj = Series(range(4), index=['d','a','b','c'])
obj.sort_index()
#DataFrame
frame = DataFrame(np.arange(8).reshape((2,4)), index=['three','one'], columns=['d','a','b','c'])
frame.sort_index() #对行索引排序
frame.sort_index(axis=1)#对列索引排序
#按照数值进行排序
#Series
obj = Series([3,4,2,5])
obj.sort_values()
#DataFrame
frame = DataFrame({'b':[4,7,-3,2],'a':[0,1,0,1]})
frame.sort_values(by=['b'])  #根据b列来对行索引排序
frame = DataFrame(np.arange(8).reshape((2,4)), index=['three','one'], columns=['d','a','b','c'])
frame.sort_values(by=['one'],axis=1, ascending=False)#根据one行的降序排列对列索引进行重排
#排名函数
obj = Series([7,-5,7,4,2,0,4])
obj.rank() #默认情况下，相等的值排名一样
obj.rank(method='first') #相等的值，先出现的排名靠前(升序)或靠后(降序)
obj.rank(method='max', ascending=False) #先按照first方法排序，然后取相同值最大排名作为相同值的最终排名
```

**rank方法的method属性的取值:**

- average  默认:在相等分组中，为各个值分配平均排名
- min 使用整个分组的最小排名
- max 使用整个相等分组的最大排名
- first 按值在原始数据中出现的顺序分配排名

#### 带有重复值的轴索引

```python
obj = Series(range(5), index=['a','a','b','b','c'])
obj
#判断索引是否是唯一的
obj.index.is_unique
'''False'''
```

## 汇总和计算描述统计

> 对Series和DataFrame对象调用sum、mean等数学统计方法时会忽略NaN。通过设置skipna=False来防止忽略NaN。

```python
df = DataFrame([[1,np.nan],[7, -4],[np.nan, np.nan],[0,-1]],index=['a','b','c','d'],columns=['one','two'])
df.sum() #求每一列的总值
df.sum(axis=1) #求每一行的总值
df.mean(axis=1, skipna=False) 
```

### 描述和汇总统计

- count  非NaN值的数量
- describe 针对Series或各DataFrame列计算汇总统计
- min、max 计算最小值和最大值
- argmin、argmax 计算能够获取到最小值和最大值的索引位置(整数)
- idxmin, idxmax  计算能够获取到最小值和最大值的索引值
- quantile  计算样本的分位数(0到1)
- sum  值的总和
- mean 值的平均数
- median 值的算术中位数，50%分位数
- mad 根据平均值计算平均绝对离差
- var 样本值的方差
- std 样本值的标准差
- skew 样本值的偏度(三阶矩)
- kurt 样本值的峰度(四阶矩)
- cumsum  样本值的累计和
- cummin、cummax 样本值的累计最大值和累计最小值
- cumprob 样本值的累计积
- diff 计算一阶差分(对时间序列很有用)
- pct_change 计算百分数变化

### 相关系数与协方差

- pct_change——百分数变化

- Series.corr——计算两个Series中重叠的、非NaNDE 、按索引对齐的值的相关系数

  ```python
  returns.MSFT.corr(returns.IBM)
  ```

- Series.cov——计算协方差

- DataFrame.corr, DataFrame.cov以DF的形式返回完整的相关系数或协方差矩阵

- DataFrame.corrwith——计算其列或行跟另一个Series或DataFrame之间的相关系数。(传入axis=1即可按行进行计算)

  ```python
  #传入Series返回一个相关系数值Series(针对各列进行计算)
  returns.corrwith(returns.IBM)
  #传入DF会计算按列名配对的相关系数。
  ```

### 唯一值、值计数以及成员资格

- Series值出现频率

  ```python
  obj = Series(['c','a','d','a','a','b','b','c','c'])
  #默认按值频率降序排列
  obj.value_counts()
  '''
  c    3
  a    3
  b    2
  d    1
  dtype: int64
  '''
  #不按照值频率排列
  pd.value_counts(obj.values, sort=False)
  '''
  a    3
  c    3
  b    2
  d    1
  dtype: int64
  '''
  #选取子集
  mask = obj.isin(['b','c'])
  obj[mask]
  '''
  0    c
  5    b
  6    b
  7    c
  8    c
  dtype: object
  '''
  #DataFrame中使用value_counts
  data = DataFrame({'Qu1':[1,3,4,3,4],'Qu2':[2,3,1,2,3],'Qu3':[1,5,2,4,4]})
  result = data.apply(pd.value_counts).fillna(0)
  ```

  **唯一值、值计数、成员资格方法:**

  - isin  ——计算一个表示"Series各值是否包含于传入的值序列中"的布尔型数组
  - unique——计算Series中的唯一值数组，按发现的顺序返回
  - value_counts——返回一个Series，其索引为唯一值，其值为频率，按计数值降序排列

## 处理缺失数据

#### NaN处理方法

- dropna——根据各标签的值中是否存在缺失数据对轴标签进行过滤，可通过阈值调节对缺失值的容忍度
- fillna——用指定值或插值方法(如ffill或bfill)填充缺失数据
- isnull——返回一个含有布尔值的对象，这些布尔值表示哪些值是缺失值/NA，该对象的类型与源类型一样
- notnull——isnull的否定式

#### 滤除缺失数据

```python
from numpy import nan as NA
#Series
data = Series([1,NA,3.5,NA,7])
data.dropna()
#DataFrame
data = DataFrame([[1,6,3],[1,NA,NA],[NA,NA,NA],[NA,6,3]])
#默认删除包含NaN的行
cleaned = data.dropna()
#删除全部为NaN的行
data.dropna(how='all')
#丢弃列
cleaned = data.dropna(axis=1)
df = DataFrame(np.random.randn(7,3))
df.loc[:4,1] = NA
df.loc[:2,2] = NA
#删除那些非NaN值的数量小于thresh值的行
df.dropna(thresh = 3)
```

#### 填充缺失值

> 只能对列操作，只能每一列上传递填充操作。因为只有同一列的数据才能保证类型一致。而对行操作无法保证类型一致。

```python
#最基本的直接把所有的NaN值替换为某个值
df.fillna(0)
#分别对不同的列采用不同的替换值
df.fillna({1:0.5,3:-1})

df = DataFrame(np.random.randn(6,3))
df.loc[2:,1] = NA
df.loc[4:,2] = NA
#使用前面最接近的一个非NaN值来替换所有随后的那些NaN值
df.fillna(method='ffill')
#限制最大的替换数量
df.fillna(method='ffill', limit=2)
#填充数可以是列的中位数，众数，平均数
df.fillna({1:df[1].mean})
#在列上传播 axis=1,或者使用apply方法来对指定行或列进行缺失值填充
df.fillna(method='ffill', axis=1)
```

- value——用于填充缺失值的标量值或字典对象
- method——插值方式。如果函数调用时未指定其他参数的话，默认为'ffill'
- axis——带填充的轴
- inplace——修改调用者对象而不产生副本
- limit——（对于前向和后向填充）可以连续填充的最大数量

## 层次化索引

> 层次化索引使你能在一个轴上拥有多个（两个以上）索引级别。抽象点说，它使你能以低维度形式处理高维度数据。

```python
data = Series(np.random.randn(10), index=[['a','a','a','b','b','b','c','c','d','d'],[1,2,3,1,2,3,1,2,2,3]])
data.index
data['b']
#使用loc读取多个一级索引时，需要以及索引是有序的。切片也同理
data= data.sort_index()
data.loc[['a','b']]
data['a':'c']
#同时指定一级索引和二级索引的方式
data[:,2]
#使用unstack方法时，需要保证不存在duplicate冲突
data.unstack()  #unstack 和stack互为逆操作
frame = DataFrame(np.arange(12).reshape((4,3)),
                 index=[['a','a','b','b'],[1,2,1,2]],
                 columns=[['Ohio','Ohio','Colorado'],
                         ['Green','Red','Green']])
#对每级索引设置一个名称
frame.index.names = ['key1','key2']
frame.columns.names=['state','color']
#选取列分组
frame['Ohio']
frame['Ohio','Green']
#选取行分组
frame.loc['a']
frame.loc['a',1]
#单独创建MultiIndex
pd.MultiIndex.from_arrays([['Ohio','Ohio','Colorado'],['Green','Red','Green']], names=['state','color'])
```

### 重排分级顺序

> 重新调整某条轴上各级别的顺序，或根据指定级别上的值对数据进行排序。

```python
#两个不同级别的索引进行互换
frame.swaplevel('key1','key2')
#按照指定的索引的有序排列对数据进行重排
frame.sort_index(level=1)
frame.sort_index(level='key2') #跟上面的语句效果一样，一个是编号，一个是名称
#列的层级索引排序
frame.sort_index(level='color', axis=1 ,ascending=False) #降序
frame.sort_index(level=1, axis=1, ascending=False) #等效于上面的语句
```

### 根据级别汇总统计

> 通过level选项指定在某条轴上求和的级别。再以上面那个DataFrame为例，我们可以根据行或列上的级别来进行求和。

```python
frame.sum(level='key2')
frame.sum(level='color',axis=1)
```

### 使用DataFrame的列

> 将DataFrame的一个或多个列当作行索引来用，或者可能希望将行索引变成DataFrame的列。

```python
#把c,d列作为索引，同时把这两列从列数据中删除
frame2 = frame.set_index(['c','d'])
#同上，但是保留这两列
frame2 = frame.set_index(['c','d'],drop=False)
#set_index的逆操作reset_index
frame2.reset_index()
```

## 其他有关pandas的话题

### 整数索引

```python
ser = Series(np.arange(3.))
#当索引为整数的时候，下面的语法错误
ser[-1]
#但是切片可以使用
ser3 = Series(range(4), index=[-5,1,3,6])
ser3[-5:6]
ser3[-5:3] #结果一样

ser2 = Series(np.arange(3.), index=['a','b','c'])
#如果是非整数索引，则可以
ser2[-1]
```

### 面板数据

<略>

