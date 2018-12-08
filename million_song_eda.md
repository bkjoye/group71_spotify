---
title: Million Song EDA
nav_include: 2
---


```python
import numpy as np
import pandas as pd

import math
from scipy.special import gamma

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

from IPython.display import display

import os
import re
import json
```




```python
def process_file(file_name, data_dict):
    with open(file_name) as json_data:
        data = json.load(json_data)
    tag_length = len(data['tags'])
    similars_length = len(data['similars'])
    key_str = file_name.split('/')[-1][:-5]
    data_dict[key_str] = data
    return data_dict
```




```python
def generate_file_list(directory):
    file_list = []
    #this loop properly gets all files in the directory
    for directory, sub_dirs, files in os.walk(directory):
        for name in files:
            if name[-4:] == 'json':
                file_list.append(directory + '/' + name)
    return file_list
```




```python
def parse_data(data):
    similars_list = []
    tags_list = []
    songs_list = []
    for key, song in data.items():
        songs_list.append([song['artist'], song['timestamp'], song['track_id'], song['title']])
        for similar_list in song['similars']:
            similars_list.append([key]+similar_list)
        for tag_list in song['tags']:
            tags_list.append([key, re.sub('[^a-z0-9 ]+','',tag_list[0].lower()), tag_list[1]])
        
    similars_df = pd.DataFrame(similars_list, columns=['track_id1', 'track_id2', 'similarity'])
    tags_df = pd.DataFrame(tags_list, columns=['track_id', 'tag', 'strength'])
    songs_df = pd.DataFrame(songs_list, columns=['artist', 'timestamp', 'track_id', 'title'])
    
    return similars_df, tags_df, songs_df
```




```python
def process_data(directory_in, save_to_disk=True, 
                 similars_file_out='data/similars_df.json', 
                 tags_file_out='data/tags_df.json', 
                 songs_file_out='data/songs_df.json'):
    #retrieve list of all json files in directory and subdirectories
    print('Generating File List')
    file_list = generate_file_list(directory_in)
    #extract data from json files into dict
    data= {}
    print('Reading data from files')
    for name in file_list:
        data = process_file(name, data)
    #parse data into three separate dataframes
    print('Putting Data into dataframes')
    similars_df, tags_df, songs_df = parse_data(data)
    
    if save_to_disk == True:
        #save dataframes for later use
        print('Saving data to disk')
        similars_df.to_json(similars_file_out)
        tags_df.to_json(tags_file_out)
        songs_df.to_json(songs_file_out)
    return similars_df, tags_df, songs_df
```




```python
similars_df, tags_df, songs_df = \
process_data('data/lastfm_train/', True, 'data/similars_train.json',\
             'data/tags_train.json', 'data/songs_train.json')
```


    Generating File List
    Reading data from files
    Putting Data into dataframes




```python
tags_df = pd.read_json('data/tags_train.json')
```




```python
display(tags_df.head())
display(tags_df.shape)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>tag</th>
      <th>strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAAAAK128F9318786</td>
      <td>alternative rock</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAAAAK128F9318786</td>
      <td>rock</td>
      <td>60</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TRAAAAW128F429D538</td>
      <td>hieroglyiphics</td>
      <td>100</td>
    </tr>
    <tr>
      <th>100</th>
      <td>TRAAAED128E0783FAB</td>
      <td>jazz vocal 2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>TRAABVM128F92CA9DC</td>
      <td>love</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>



    (7671133, 3)




```python
tags_per_song = tags_df['track_id'].value_counts()
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(tags_per_song)
ax.set_title('Number of Tags Per Song')
ax.set_ylabel('Number of Songs')
ax.set_xlabel('Number of Tags')
plt.show()
```



![png](million_song_eda_files/million_song_eda_8_0.png)




```python
songs_df = pd.read_json('data/songs_train.json')
```




```python
display(songs_df.head())
display(songs_df.shape)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>timestamp</th>
      <th>track_id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelitas Way</td>
      <td>2011-08-15 09:59:32.436152</td>
      <td>TRAAAAK128F9318786</td>
      <td>Scream</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Western Addiction</td>
      <td>2011-08-12 13:00:44.771968</td>
      <td>TRAAAAV128F421A322</td>
      <td>A Poor Recipe For Civic Cohesion</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Son Kite</td>
      <td>2011-08-10 19:36:13.851544</td>
      <td>TRAAAEM128F93347B9</td>
      <td>Game &amp; Watch</td>
    </tr>
    <tr>
      <th>100</th>
      <td>Lost Immigrants</td>
      <td>2011-08-02 09:37:00.958971</td>
      <td>TRAACEI128F930C60E</td>
      <td>Memories &amp; Rust</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>The Irish Tenors</td>
      <td>2011-08-03 03:38:35.708526</td>
      <td>TRAATLC12903D0172B</td>
      <td>Mountains Of Mourne</td>
    </tr>
  </tbody>
</table>
</div>



    (839122, 4)




```python
#not enough ram to execute, need to find alternate solution.
#Dataframe is loaded after running the process_data function.
similars_df = pd.read_json('data/similars_train.json')
display(similars_df.head())
```




```python
fig, ax = plt.subplots(figsize=(10,5))
tag_counts = tags_df['tag'].value_counts()
tag_counts_gt5 = tag_counts[tag_counts>5]
ax.plot(np.arange(tag_counts.shape[0])+1, tag_counts)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title('Frequency of Tag Usage')
ax.set_xlabel('Tag Popularity')
ax.set_ylabel('Number of Appearances')
# ax[0].plot(np.arange(tag_counts.shape[0])+1, tag_counts)
# ax[0].set_yscale('log')
# ax[0].set_xscale('log')
# ax[0].set_title('Frequency of Tag Usage')
# ax[0].set_xlabel('Tag Popularity')
# ax[0].set_ylabel('Number of Appearances')
# ax[1].plot(np.arange(tag_counts_gt5.shape[0]), tag_counts_gt5)
# ax[1].set_yscale('log')
# ax[1].set_xscale('log')
# ax[1].set_title('Frequency of Tag Usage')
# ax[1].set_xlabel('Tag Popularity')
# ax[1].set_ylabel('Number of Appearances')
plt.show()
```



![png](million_song_eda_files/million_song_eda_12_0.png)




```python
data = [[i, w] for i, w in tag_counts[0:10].items()]
```




```python
pd.DataFrame(data, columns=['Tag', 'Usage Count'], index=np.arange(1,11))
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tag</th>
      <th>Usage Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>rock</td>
      <td>91222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pop</td>
      <td>61775</td>
    </tr>
    <tr>
      <th>3</th>
      <td>alternative</td>
      <td>50568</td>
    </tr>
    <tr>
      <th>4</th>
      <td>indie</td>
      <td>43037</td>
    </tr>
    <tr>
      <th>5</th>
      <td>electronic</td>
      <td>40525</td>
    </tr>
    <tr>
      <th>6</th>
      <td>female vocalists</td>
      <td>37804</td>
    </tr>
    <tr>
      <th>7</th>
      <td>favorites</td>
      <td>37029</td>
    </tr>
    <tr>
      <th>8</th>
      <td>love</td>
      <td>31497</td>
    </tr>
    <tr>
      <th>9</th>
      <td>dance</td>
      <td>29495</td>
    </tr>
    <tr>
      <th>10</th>
      <td>00s</td>
      <td>28699</td>
    </tr>
  </tbody>
</table>
</div>


