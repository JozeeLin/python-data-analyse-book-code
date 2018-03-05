# Pandas DataFrame引言

[TOC]

## DataFrame用于将数据表示为一个表格

```
df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]})
print df
```

- DataFrame 对某一列的所有取值分别进行统计

  DataFrame.value_counts()

  `frame = DataFrame()`

  `counts = frame[columnName].value_counts()`

- 填充缺失值和替换指定值

  `frame = DataFrame()`

  `//填充缺失值`

  `frame[columnsName].fillna('特定值')`

  `frame[frame[columnName] == 指定值] = 新的值`

- 判断某一列中字符串取值是否包含指定字符串

  frame['a'].str.contains(指定字符串)

- 根据条件判断来重新赋值为不同的值

  np.where(cframe['a'].str.contains('Windows'), 'windows', 'Not windows')

- DataFrame数据合并

  pandas.merge(df1, df2)

- 读取指定索引的行inde

  data.iloc[索引]

- DataFrame聚合操作pivot_table

  > 按照性别计算每部电影的平均得分
  >
  > mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')

- 对数据分组进行，并进行统计

  `data.groupby('title').size()`

- 按照指定列进行排序

  `DataFrame.sort_index(by='F', ascending=False)`

- 对排序结果反序

  sorted_dataframe<u>**[::-1]**</u>

- 对Series排序

  Series.sort_values(ascending=False)降序

  ​