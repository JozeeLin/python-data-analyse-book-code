# 附录A Python语言精要

[TOC]

## 基础知识

### 万物皆对象

> python语言的一个重要特点就是其对象模型的一致性。python解释器中的任何数值、字符串、数据结构、函数、类、模块等都待在它们自己的“盒子”里，而这个“盒子”也就是python对象。每个对象都有一个与之相关的类型(比如字符串或函数)以及内部数据。在实际工作当中，这使得python语言变得非常灵活，因为即使是函数也能被当做其他对象那样处理。

### 注释

> 以'#'开头的行为注释，会被解释器忽略

### 函数调用和对象方法调用

```python
#伪代码！！！
#带有0个或多个参数的函数调用,且把返回值赋值给其他变量
result = f(x,y,z)
g()
#方法调用，类似于函数，但是他属于对象内部的函数
obj.some_method(x,y,z)
#接受位置参数，也可以接受关键字参数
result = f(a,b,c,e='foo')
```

### 变量和按引用传递

```python
#对变量赋值，就是在创建等号右侧对象的一个引用
a = [1,2,3]
b = a
#可以验证,修改a，b也会被修改
a.append(4)
print b
```

> 赋值(assignment)操作也叫做绑定(binding)。因为赋值操作其实就是将一个名称绑定到一个对象上。已经赋过值的变量名有时也被称为已绑定变量。

```python
#参数传递也只是传入了一个引用，不会发生复制
def append_element(some_list, element):
    some_list.append(element)

data = [1,2,3]
append_element(data, 4)
print data
```

### 动态引用，强类型

```python
#python对象引用没有与之关联的类型信息,a只是一个名称，可以绑定在任何类型的对象上
a = 5
print type(a)
a = 'foo'
print type(a)

#类型的隐式转换
#整型和浮点型之间运算可以自动类型转换
a = 4.5
b=2
print a/b
#整型和字符串类型之间进行运算不能自动类型转换，会报错
'5'+5

#判断对象的类型
a = 5
#判断对象a是否是整型
print isinstance(a,int)
#类型信息可以使用元组表示
print isinstance(a,(int, float))
b=4.5
print isinstance(b, (int,float))
```

### 属性和方法

> python对象通常都既有属性(attribute,即存储在该对象'内部'的其他python对象)又有方法(method,与该对象相关的能够访问其内部数据的函数)。访问方式为obj.attribute_name

```python
#使用点运算符来访问
a = 'foo'
a.endswith

#使用getattr函数来访问
#getattr、hasattr、setattr函数在编写通用的、可复用的代码时很有用
getattr(a,'split')
```

### 鸭子"类型"

> 只要一个对象实现了迭代器协议(iterator protocol),你就可以确认它是可迭代的。这也意味着这个对象拥有一个**iter**魔术方法。还有比较便捷的验证方法。使用iter函数，如果没有引发TypeError那么就表示它是一个可迭代对象。

```python
#判断一个对象是否是可迭代对象，类型判断
def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False
```

### 引入

```python
#创建一个python文件，some_module.py,内容如下
PI=3.14159
def f(x):
    return x+2

def g(a,b):
    return a+b

#在同一个目录下创建另外一个文件，在文件中可以引用some_module.py的内容
import some_module
s = 2
result = some_module.f(s)
pi = some_module.PI

#还可以这样引入
from some_module import f,g,PI
result = g(5,PI)

import some_module as sm
from some_module import PI as pi,g as gf
r1 = sm.f(pi)
r2 = gf(6,pi)
```

### 二元运算符和比较运算符

```python
5-7
12+21.5
5<=2
#判断两个引用是否指向同一个对象，使用is关键字。
#判断两个引用是否不是指向同一个对象，使用is not
a = [1,2,3]
b = a
#list函数会创建新的列表
c = list(a)
print  a is b
print a is not c

#==运算符用于比较两个对象的值
a == c

#通常使用is、is not用于判断对象是否为None
a = None
a is None
```

**二元运算符：**

- a+b——a加b
- a-b——a减b
- a*b——a乘以b
- a/b——a除以b
- a//b——a除以b后向下圆整，丢弃小数部分
- a**b——a的b次方
- a&b——如果a和b均为True，则结果为True。对于整数，执行按位与操作
- a|b——如果a和b任何一个为True，则结果为True。对于整数，执行按位或操作
- a^b——对于布尔值，如果a或b为True(但不同时为True)，则结果为True。对于整数，执行按位异或操作
- a==b——如果a等于b，则结果为True
- a!=b——如果a不等于b，则结果为True
- a<=b、a<b——如果a小于等于(小于)b，则结果为True
- a>b、a>=b——如果a大于(或大于等于)b,则结果为True
- a is b——如果引用a和b指向同一个python对象，则结果为True
- a is not b——如果引用a和b指向不同的python对象，则结果为True

### 严格与懒惰

> 通常情况下，python数值表达式是立即计算出结果的。但是可以通过迭代器、生成器这些技术实现延迟计算。类似于函数编程Haskell的延时计算思想

### 可变和不可变的对象

```python
#列表、字典、numpy数组是可变对象
a_list = ['foo',1,2,[4,5]]
a_list[2] = (3,4)
a_list
#字符串和元组为不可变的
a_tuple = (2,3,4,(4,5))
a_tuple[2] = 'foo'
```

> "可以修改某个对象"这种行为在编程中称为“副作用(side effect)“,任何副作用都应该通过该函数的文档或注释明确地告知用户。即使可以使用可变对象，也建议尽量避免副作用且注重不变性(immutability)

### 标量类型

**标准的python标量类型**

- None——python的‘null'值（None只存在一个实例对象）
- str——字符串类型。python 2.x中只有ASCII值，而python3中则是Unicode
- unicode——unicode字符串类型
- float——双精度(64位)浮点书。注意，这里没有专门的double类型
- bool——True或False
- int——有符号整数，其最大值由平台决定
- long——任意精度的有符号整数。大的int值会被自动转换为long

#### 数值类型

```python
#python3,整数相除，除不尽，产生浮点数
#3/2=1.5
#python 2.x返回整数
#3/2=1
#python 2.x通过from __future__ import division,可以实现python3的效果
from __future__ import division
print 3/2
#python2.x也可以显示把其中一个数转成浮点型
print 3/float(2)

3//2

#复数的表示和运算
cval = 1+2j
cval*(1-2j)
```

#### 字符串

```python
a = 'one way of writing a string'
b = 'another way'
#带有换行符的多行字符串，使用三重引号('''或""")
c = """
this is a longer string that
spans multiple lines
"""
#字符串是不可变的。要修改字符串就只能创建一个新的
a = 'this is a string'
a[0] = 'f'
#修改字符串可以先把它转成可迭代的对象类型，修改完再还原回字符串
a = list(a)
a[0] = 'f'
a = ''.join(a)
print a
#转义符——反斜杠\
s = '12\\34'
print s
#r表示所有字符应该按照原来的样子解释，不需要转义符，且转义符无效
s = r'this\has\on\special\characters'
print s
#两个字符串加起来会产生一个新字符串
a = 'this is the first half'
b = 'and this is the second half'
a+b
#字符串格式化。以%开头且后面跟着一个或多个格式字符的字符串是需要插入值的目标
template = '%.2f %s are worth $%d'
template % (4.556, 'Argentine Pesos', 1)
```

#### 布尔值

```python
#and、or关键字用于连接布尔值
print True and False
print True or False
print True and True
print False and False
a = [1,2,3]
if a:
    print 'I found something!'

b = []
if not b:
    print 'Empty!'
    
#使用bool函数可以知道某个对象究竟会被强制转换成哪个布尔值
bool([]),bool([1,2,3])
bool('Hello,world!'),bool('')
bool(0),bool(1)
```

#### 类型转换

```python
s = '3.14159'
fval = float(s)
print type(fval)
print int(fval)
print bool(fval)
print bool(0)
```

#### None

```python
#None是Python的空值类型。如果一个函数没有显式地返回值，则隐式返回None
a = None
a is None
b = 5
b is not None
#None还是函数可选参数的一种常用默认值
def add_and_maybe_multiply(a,b,c=None):
    result = a+b
    if c is not None:
        result = result + c
    return result
```

#### 日期和时间

```python
from datetime import datetime,date,time
dt = datetime(2011,10,29,20,30,21)
print dt.day
print dt.minute
print dt.date()
print dt.time()
#strftime 用于将datetime格式化为字符串
dt.strftime('%m/%d/%Y %H:%M')
#字符串可以通过strptime函数转换(解析)为datetime对象
datetime.strptime('20091031','%Y%m%d')
#datetime之差，datetime.timedelta类型
dt2 = datetime(2011,11,15,22,30)
delta = dt2 - dt
print delta
print type(delta)
#datetime和timedelta之和
dt + delta
```

### 控制流

```python
#if,elif,else,条件语句
x = 0
if x < 0:
    print "it's negative"
elif x == 0:
    print "Equal to zero"
elif 0<x<5:
    print "Positive but smaller than 5"
else:
    print "Positive and larger than or equal to 5"
    
#对于and或or组成的复合条件，复合条件是按从左到右的顺序求值的，而且是短路型
a = 5;b = 7
c = 8;d = 4
if a<b or c>d:
    print 'Made it'
 
```

#### for循环

> for循环用于对集合(比如列表或元组)或迭代器进行迭代。

```python
#continue关键字用于使for循环提前进入下一次迭代
sequence = [1,2,None,4,None,5]
total = 0
for value in sequence:
    if value is None:
        continue
    total += value
    
#break关键字用于使for循环完全退出
sequence = [1,2,0,4,6,5,2,1]
total_until_5 = 0
for value in sequence:
    if value == 5:
        break
    total_until_5 += value
```

#### while循环

```python
x = 256
total = 0
while x>0:
    if total>500:
        break
    total += x
    x = x // 2
```

#### pass

> pass是python中的"空操作"语句。它可以被用在那些没有任何功能的代码块中

```python
if x<0:
    print 'negative!'
elif x==0:
    #TODO:在这里放点代码
    pass
else:
    print 'positive'
    
def f(x,y,z):
    #TODO: 实现这个函数！
    pass
```

#### 异常处理

> 优雅地处理python错误或异常是构建健壮程序的重要环节。

```python
float('12.345')
#编写一个在出错时能优雅地返回输入参数的改进版float函数。
def attempt_float(x):
    try:
        return float(x)
    except:
        return x
    
attempt_float('1.23')
attempt_float('something')
#只希望处理ValueError
def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        return x
    
attempt_float((1,2))
#同时处理ValueError、TypeError
def attempt_float(x):
    try:
        return float(x)
    except(TypeError, ValueError):
        return x
    
#只希望有一段代码，不管try块代码成功与否都能被执行。使用finally即可达到这个目的
# f=open(path,'w')
#try:
#    write_to_file(f)
#finally:
#    f.close()
#让某些代码只在try块成功时执行。使用else即可
#f = open(path,'w')
#try:
#    write_to_file(f)
#except:
#    print 'Failed'
#else:
#    print 'Succeeded'
#finally:
#    f.close()
```

#### range和xrange

```python
#range函数用于产生一组间隔平均的整数
range(10)
#指定起始值、结束值以及步长等信息
range(0,20,2)
#range常用于按索引对序列进行迭代
seq = [1,2,3,4]
for i in range(len(seq)):
    val = seq[i]
    
#对于非常长的范围，建议使用xrange，
#它不会预先产生所有的值并将它们保存到列表中。
#返回一个用于逐个产生整数的迭代器。
sum = 0
for i in xrange(10000):
    #%是求模运算符
    if x%3==0 or x%5==0:
        sum += i
```

> 注意:python3中，range始终返回迭代器，xrange可以丢弃

#### 三元表达式

> 语法：value = true-expr if condition else false-expr

```python
x=5
'Non-negative' if x>=0 else 'Negative'
```

### 数据结构和序列

#### 元组

> 元组是一种一维的、定长的、不可变的python对象序列。

```python
tup = 4,5,6
tup
nested_tup = (4,5,6),(7,8)
nested_tup
tuple([4,0,2])
tup = tuple('string')
tup
tup[0]
#存储在元组中的对象本身不可变，但创建完毕，存放在各个插槽中的对象就不能再被修改
tup = tuple(['foo',[1,2],True])
tup[2] = False
tup[1] = [1,2,3]
tup[1].append(3)
tup
#元组拼接
(4,None,'foo')+(6,0)+('bar',)
('foo','bar')*4
```

#### 元组拆包

```python
tup = (4,5,6)
a,b,c=tup
b
#嵌套元组被拆包
tup = 4,5,(6,7)
a,b,(c,d) = tup
d
#利用该功能很便利的交换变量名
tmp = a
a = b
b = tmp

#下面的python风格代码等价于上面的C语言风格代码
b,a = a,b
#变量拆包功能常用于对由元组或列表组成的序列进行迭代
seq = [(1,2,3),(4,5,6),(7,8,9)]
for a,b,c in seq:
    pass
```

#### 元组方法

```python
#统计指定值出现的次数
a = (1,2,2,2,3,4,2)
a.count(2)
```

#### 列表

```python
a_list = [2,3,7,None]
tup = ['foo','bar','baz']
b_list = list(tup)
b_list
b_list[1] = 'peekaboo'
b_list
```

#### 添加和移除元素

```python
#append在列表的末尾添加元素
b_list.append('dwarf')
b_list
#把元素插入到列表的指定位置，计算量比append大
b_list.insert(1,'red')
b_list
#insert的逆运算是pop，用于移除并返回指定索引处的元素
b_list.pop(2)
b_list
b_list.append('foo')
b_list
#remove用于按值删除元素，找到第一个符合要求的值然后将其从列表中删除
b_list.remove('foo')
b_list
#通过关键字in,判断列表中是否含有某个值
'dwarf' in b_list
```

> 注意:判断列表中是否含有某个值的操作比字典(dict)和集合(set)慢得多，因为python会对列表中的值进行线性扫描。而另外两个(基于哈希表)则可以瞬间完成判断

#### 合并列表

```python
#使用+号将两个列表加起来
[4,None,'foo']+[7,8,(2,3)]
#使用extend对已存在列表一次性添加多个元素
x = [4,None,'foo']
x.extend([7,8,(2,3)])
x
```

> 注意：列表的加法合并是很浪费资源的操作，因为它需要创建新的列表，并将所有对象复制过去。而extend将元素附加到现有列表。因此加法和并比extend方法慢很多

#### 排序

```python
#调用列表的sort方法可以实现就地排序
a = [7,2,5,1,3]
a.sort()
#sort的选项。次要排序键，即一个能够产生可用于排序的值的函数。
#通过长度对一组字符串进行排序
b = ['saw','small','He','foxes','six']
b.sort(key=len)
b
```

#### 二分搜索及维护有序列表

> 内置的bisect模块实现了二分查找以及对有序列表的插入操作。bisect.bisect可以找出新元素应该被插入到哪个位置才能保持原列表的有序性。而bisect.insort则确实地将新元素插入到那个位置上去。

```python
import bisect
c = [1,2,2,2,3,4,7]
bisect.bisect(c,2)
bisect.bisect(c,5)
```

> bisect模块的函数不会判断原列表是否是有序的，因为这样做的开销太大了。因此，将它们用于无序列表虽然不会报错，但可能会导致不正确的结果

#### 切片

> 通过切片标记法，你可以选取序列类型的子集。

```python
seq = [7,2,3,7,5,6,0,1]
seq[1:5]
seq[3:4] = [6,3]
seq[:5]
seq[3:]
seq[-4:]
seq[-6:-2]
#在第二个冒号后面加上步长(step)
seq[::2]
#-1实现列表或元组的反序
seq[::-1]
```

### 内置的序列函数

#### enumerate

```python
#对一个序列进行迭代时，常常需要跟踪当前项的索引
i = 0
collection = [1,2,3]
for value in collection:
    pass
    i += 1
    
#使用enumerate可以同时跟踪序列的索引和值
for i,value in enumerate(collection):
    #TODO
    pass
#求取一个将序列值映射到其所在位置的字典
some_list = ['foo','bar','baz']
mapping = dict((v,i) for i,v in enumerate(some_list))
mapping
```

#### sorted

> sorted函数可以将任何序列返回为一个新的有序列表

```python
sorted([7,1,2,6,0,3,2])
sorted('horse race')
#常常将sorted和set结合起来使用以得到一个由序列中的唯一元素组成的有序列表
sorted(set('this is just some string'))
```

#### zip

> 用于将多个序列(列表、元组等)中的元素"配对"，从而产生一个新的元组列表

```python
seq1 = ['foo','bar','baz']
seq2 = ['one','two','three']
zip(seq1,seq2)
#接受任意数量的序列，最终的得到的元组数量由最短的序列决定
seq3 = [False,True]
zip(seq1,seq2,seq3)
for i,(a,b) in enumerate(zip(seq1,seq2)):
    print('%d: %s,%s' % (i,a,b))
    
pitchers = [('Nolan','Ryan'),('Roger','Clemens'),('Schilling','Curt')]
first_names,last_names = zip(*pitchers)
first_names
last_names
```

#### reversed

> 用于按逆序迭代序列中的元素

```python
list(reversed(range(10)))
```

### 字典

```python
empty_dict = {}
d1 = {'a':'some value','b':[1,2,3,4]}
d1
#访问、插入、设置元素的语法跟元素、列表一样
d1[7] = 'an interger'
d1['b']
#判断字典中是否存在某个键
'b' in d1
d1[5] = 'some value'
d1['dummy'] = 'another value'
#使用del关键字或pop方法删除值
del d1[5]
print d1
ret = d1.pop('dummy')
print ret
print d1
d1.keys()
d1.values()
d1.items()
#update 方法，一个字典可以被合并到另一个字典中去
d1.update({'b':'foo','c':12})
```

> 注意：python3中，dict.keys()和dict.keys()会返回迭代器而不是列表

#### 从序列类型创建字典

```python
#字典本质上是一个二元元组集。所以可以用dict类型函数直接处理二元元组列表
mapping = dict(zip(range(5),reversed(range(5))))
```

#### 默认值

```python
#根据首字母对一组单词进行分类并最终产生一个由列表组成的字典
words = ['apple','bat','bar','atom','book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
#使用setdefault可以替代上面的if-else语句
by_letter1 = {}
for word in words:
    letter = word[0]
    by_letter1.setdefault(letter, []).append(word)
    
#内置的collections模块的defaultdict类，是过程更简单
#传入一个类型或函数(用于生成字典各插槽所使用的默认值)即可创建出一个defaultdict
from collections import defaultdict
by_letter = defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)

#defaultdict初始化器只需要一个可调用对象。
#比如，想要将默认值设置为4,只需要传入一个能够返回4的函数即可
counts = defaultdict(lambda: 4)
```

#### 字典键的有效类型

> 字典的键是不可变对象

```python
a = {}
a[['a']] = 1
#通过hash函数，你可以判断某个对象是否是可哈希的(即可以用作字典的键)
hash('string')
hash((1,2,3))
hash([1,2])
#如果要将列表当作键，最简单的办法就是将其变成元组
d = {}
d[tuple([1,2,3])] = 5
```

### 集合

> 集合(set)是由唯一元素组成的无序集。可以将其看做只有键没有值的字典

```python
set([1,2,3])
{2,2,3,4,5}
a = {1,2,3,4,5}
b = {3,4,5,6,6}
#求并集
a|b
#求交集
a&b
#求差集
a-b
#求对称集，异或
a^b
#判断一个集合是否是另一个集合的子集(原集合包含于新集合)或超集(原集合包含新集合)
a_set = {1,2,3,4,5}
{1,2,3}.issubset(a_set)
#超集合
a_set.issuperset({1,2,3})
{1,2,3} == {1,3,2}
```

**python的集合运算**

- a.add(x)——N/A——将元素x添加到集合a
- a.remove(x)——N/A——将元素x从集合a中删除
- a.union(b)——a|b——a和b全部的唯一元素
- a.intersection(b)——a&b——a和b都有的元素
- a.difference(b)——a-b——a中不属于b的元素
- a.symmetric_difference(b)——a^b——a或b中不同时属于a和b的元素
- a.issubset(b)——N/A——如果a的全部元素都包含于b，则为True
- a.issuperset(b)——N/A——如果b的全部元素都包含于a，则为True
- a.isdisjoint(b)——N/A——如果a和b没有公共元素，则为True

### 列表、集合以及字典的推导式

> 基本形式：[expr for val in collection if condition]
> 相当于:
> result = []
> for val in collection:
> ----if condition:
> --------result.append(expr)

```python
strings = ['a','as','bat','car','dove','python']
[x.upper() for x in strings if len(x)>2]
```

> 字典推导式形式如下:
> dict_comp = {key_expr: value-expr for value in collection if condition}
>
> 集合推导式形式如下:
> set_comp={expr for value in collection if condition}

```python
unique_lengths = {len(x) for x in strings}
unique_lengths
loc_mapping = {val: index for index,val in enumerate(strings)}
loc_mapping
loc_mapping = dict((val,idx) for idx,val in enumerate(strings))
loc_mapping
```

#### 嵌套列表推导式

```python
all_data = [['Tom','Billy','Jefferson','Andrew','Wesley',
             'Steven','Joe'],['Susie','Casey','Jill','Ana',
                             'Eva','Jennifer','Stephanie']]
result = [name for names in all_data for name in names if name.count('e')>=2]
#将一个由整数元组构成的列表“扁平化”为一个简单的整数列表
some_tuples = [(1,2,3),(4,5,6),(7,8,9)]
flattened = [x for tup in some_tuples for x in tup]
[[x for x in tup] for tup in some_tuples]
[tup for tup in some_tuples]
```

## 函数

> 关键字参数必须位于位置参数之后

### 命名空间、作用域、以及局部函数

> 函数可以访问两种不同作用域的变量：全局(global)和局部(local)。python有一种更科学的用于描述变量作用域的名称，即命名空间。任何在函数中赋值的变量默认都是被分配到局部命名空间中的。局部命名空间是在函数被调用时创建的，函数参数会立即填入该命名空间。在函数执行完毕之后，局部命名空间就会被销毁。

```python
def func(a):
    for i in range(5):
        a.append(i)
        
a = [1,2,3]
func(a)
print a

#通过global关键字把局部函数变量转成全局变量
def bind_a_variable():
    global b
    b = [0,0,0]
bind_a_variable()
print b

#局部函数(在外层函数被调用之后才会被动态创建出来)
def outer_function(x,y,z):
    def inner_function(a,b,c):
        pass
    pass
```

### 返回多个值

```python
def f():
    a = 5
    b = 6
    c = 7
    return a,b,c
a,b,c = f()
```

### 函数亦对象

```python
states = ['   Alabama ','Georgia!', 'Georgia', 'geogia','FLOrIda',
         'south corolina##','West virginia?']

import re
def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]','',value) #移除标点符号
        result.append(value)
    return result

clean_strings(states)

def remove_punctuation(value):
    return re.sub('[!#?]','',value)

clean_ops = [str.strip,remove_punctuation,str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result

clean_strings(states, clean_ops)

map(remove_punctuation, states)
```

### 匿名函数

```python
def short_function(x):
    return x*2
equiv_anon = lambda x:x*2

def apply_to_list(some_list, f):
    return [f(x) for x in some_list]

ints = [4,0,1,5,6]
apply_to_list(ints, lambda x:x*2)

[x*2 for x in ints]

strings = ['foo','card','bar','aaa','abab']

strings.sort(key=lambda x:len(set(list(x))))

strings
```

### 闭包:返回函数的函数

> 闭包是由其他函数动态生成并返回的函数。其关键性质是，被返回的函数可以访问其创建者的局部命名空间中的变量。

```python
def make_closure(a):
    def closure():
        print('I know the secret: %d' % a)
    return closure
closure = make_closure(5)

closure()

#返回一个能够记录自身参数(曾经传入到该函数的参数)的函数
def make_watcher():
    have_been = {}
    
    def has_been_seen(x):
        if x in have_been:
            return True
        else:
            have_been[x] = True
            return False
    return has_been_seen

watcher = make_watcher()

vals = [5,6,7,1,2,3]
[watcher(x) for x in vals]

[watcher(x) for x in [1,2]]
```

> 注意：虽然可以修改任何内部状态对象，但不能绑定外层函数作用域中的变量。一个解决办法是:修改字典或列表，而不是绑定变量

```python
def make_counter():
    count = [0]
    def counter():
        #增加并返回当前的count
        count[0] += 1
        return count[0]
    return counter

counter = make_counter()
```

> 应用场景：在实际工作中，可以编写带有大量选项的非常一般化的函数，然后再组装出更简单更专门化的函数

```python
def format_and_pad(template, space):
    def formatter(x):
        return (template % x).rjust(space)
    return formatter

#创建一个始终返回15位字符串的浮点数格式化器
fmt = format_and_pad('%.4f', 15)
fmt(1.756)
```

### 扩展调用语法和*args、\**kwargs

```python
def say_hello_then_call_f(f, *args, **kwargs):
    print 'args is', args
    print 'kwargs is',kwargs
    print("Hello!Now I'm going to call %s" % f)
    return f(*args, **kwargs)

def g(x,y,z=1):
    return (x+y)/z

say_hello_then_call_f(g,1,2,z=5.)
```

### 柯里化：部分参数应用

> 柯里化(currying)是一个有趣的计算机科学术语，它指的是通过"部分参数应用"从现有函数派生出新函数的技术。

```python
def add_numbers(x,y):
    return x+y
#通过上面的函数，派生出一个新的只有一个参数的函数——add_five,它用于对其参数加5
#add_numbers的第二个参数称为"柯里化的"
add_five = lambda y:add_numbers(5,y)
from functools import partial
add_five=partial(add_numbers,5)
```

### 生成器

> 通过一种叫做迭代器协议(iterator protocal,它是一种使对象可迭代的通用方式)的方式实现。
> 迭代器是一种特殊对象，它可以在诸如for循环之类的上下文中向python解释器输送对象。

```python
some_dict = {'a':1,'b':2,'c':3}
#使用for循环迭代的时候，python解释器首先会尝试从some_dict创建一个迭代器。

for key in some_dict:
    print key,
    
dict_iterator = iter(some_dict)
dict_iterator
```

> 生成器是构造新的可迭代对象的一种简单方式。一般的函数执行之后之后返回单个值，而生成器则是以延迟的方式返回一个值序列，即每返回一个值之后暂停，直到下一个值被请求时再继续。**要创建一个生成器，只需将函数中的return替换为yeild即可。**

```python
def squares(n=10):
    print 'Generating squares from 1 to %d' % (n**2)
    for i in xrange(1,n+1):
        yield i**2
        
        
gen =squares()
gen
for x in gen:
    print x
    
def make_change(amount, coins=[1,5,10,25], hand=None):
    hand = [] if hand is None else hand
    if amount == 0:
        yield hand
    for coin in coins:
        #确保我们给出的硬币没有超过总额，且组合是唯一的
        if coin>amount or (len(hand))>0 and hand[-1]<coin:
            continue
        for result in make_change(amount-coin,coins=coins,\
                                  hand=hand+[coin]):
            yield result
            
for way in make_change(100, coins=[10,25,50]):
    print way
    
len(list(make_change(100)))
```

### 生成器表达式

> 生成器表达式是构造生成器的最简单方式。

```python
gen = (x ** 2 for x in xrange(100))
#跟上面的表达式等效
def _make_gen():
    for x in xrange(100):
        yield x**2
        
gen = _make_gen()

sum(x ** 2 for x in xrange(100))
dict((i,i**2) for i in xrange(5))
```

### itertools模块

> 标准库itertools模块中有一组用于许多常见数据算法的生成器。

```python
import itertools
first_letter = lambda x:x[0]
names = ['Alan','Adam','Wes','Will','Albert','Steven']
#groupby可以接受任何序列和一个函数。它根据函数的返回值对序列中的连续元素进行分组。
for letter,names in itertools.groupby(names, first_letter):
    print letter,list(names)
```

**一些常用的itertools函数**

- imap(func, *iterables)——内置函数map的生成器版，将func应用于参数序列的各个打包元组
- ifilter(func,iterable)——内置函数filter的生成器版,当func(x)为True时输出元素x
- combinations(iterable,k)——生成一个由iterable中所有可能的k元元组组成的序列(不考虑顺序)
- permutation(iterable,k)——生成一个由iterable中所有可能的k元元组组成的序列(考虑顺序)
- groupby(iterable[,keyfunc])——为每个唯一键生成一个(key,sub-iterator)