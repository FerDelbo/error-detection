#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from FlagEmbedding import FlagModel


# In[13]:


hospital_dirty = pd.read_csv('GEIL_Data/hospital/original/dirty.csv').astype(str)


# In[14]:


def Merge_Dict(row):
    text = {}
    for x,y in row[1:].items():
        text[x] = y
    return str(text)
hospital_dirty['merge_dict'] = hospital_dirty.apply(Merge_Dict,axis=1)


# In[ ]:


df = hospital_dirty
import pandas as pd

# 假设df是你的DataFrame
# df = pd.read_csv('your_file.csv')  # 例如，从CSV文件加载DataFrame

# 将header作为第一列添加到DataFrame中
# df['Header'] = df.columns.to_series().iloc[0]  # 将第一个header值复制到新列

# 将所有列转换为列表，包括新增的header列
all_data = df.values.flatten().tolist()  # Flatten转换DataFrame为一维数组并转换为列表

# 转换为一个不重复的列表
# unique_list = list(set(all_data))

all_data.extend(df.columns[:-1])

unique_list = list(set(all_data))

# 创建字典，键是数据，值是索引
unique_dict = {item: index for index, item in enumerate(unique_list)}

# 打印结果
print("Unique List with Header:", unique_list)
print("Unique Dictionary with Header:", unique_dict)

# 如果需要，可以删除之前添加的'Header'列
# df = df.drop(columns=['Header'])


# In[8]:


model = FlagModel('/home/yanmy/sentence_transformer_model/bge-large-en-1.5',use_fp16=True) ## 读取上一步的结果


# In[19]:


embedding = model.encode([str(a) for a in unique_list
                          ])


# In[20]:


hospital_dirty_mapping = hospital_dirty.copy()
for header in hospital_dirty_mapping.columns:
    hospital_dirty_mapping[header] = hospital_dirty_mapping[header].map(unique_dict)


# In[21]:


hospital_dirty_mapping


# In[26]:


triple_list = []
for index,row in hospital_dirty_mapping.iterrows():
    triple_subset_list = []
    index = row['merge_dict']
    for x,y in row[:-1].items():
        triple_subset_list.append([index,unique_dict[x],y])
    triple_list.append(triple_subset_list)


# In[30]:


import numpy as np
np.array(triple_list).shape


# In[32]:


np.save('PyG/hospital_graph/unique_dict.npy',unique_dict)
np.save('PyG/hospital_graph/unique_list.npy',unique_list)
np.save('PyG/hospital_graph/triple_list.npy',triple_list)
np.save('PyG/hospital_graph/embedding.npy',embedding)

