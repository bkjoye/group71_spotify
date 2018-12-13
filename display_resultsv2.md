---
title: Modelling Results
nav_include: 7
---

**Analysis Approach:**

Since our model runs took a very long time, we saved all of our metrics into dataframes to be loaded for later analysis. Each metric has six different values for each metric, based on the mean of that metric for the results of that batch. 

**Initial Approach** 

Initially we planned on aggregating the results of the batches and then comaring the different model results. However we found that the way we created our metrics gave us results with very high variability. If we were able to run the models again, we would change the metrics to be proportions of change, i.e. %change in followers vs the current method of absolute change in followers. This would normalize our data and allow us to directly compare results on playlists with large differences in number of songs and number of followers.

**Compromise Approach** 

In order to try and compare models with the metrics we already have, we combined the means of each batch along with the standard deviation of the means for each metric and compared those values. This led to some strange results, with the validation and test sets consistently outperforming the train set for each metric. However when you look at the $\pm 2\sigma $ bounds on the scores, you can see that there is significant overlap. 



```python
import sys
import datetime
import numpy as np
import pandas as pd
import string
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import gzip
import csv
import matplotlib
import matplotlib.pyplot as plt

DATA_DIR="../../../data"
```




```python
def add(r, names, df) :
    for m in names:
        r[m].append(df[m].mean())

def tonp (r, names):
    for m in names:
        r[m] = np.array(r[m])
    
def readResults(n, shortname, name):
    m_names = ['metric', 'match', 'distance', 'numf', 'diff']
    prefix = DATA_DIR + "/results/" + shortname + str(n) + "/result_" + name
    suffix = "_" + str(n) + "_10.csv.gz"
    r = {'metric':[], 'match' : [], 'distance' : [], 'numf' : [], 'diff' : [], 'metric2' : [] }
    for i in range(1, 7) :
        fullName = prefix + str(i) + suffix
        df = pd.read_csv(fullName, compression='gzip')#.drop(['Unnamed: 0'],axis=1)
        add(r, m_names, df)
        r['metric2'].append(((1.0 / df['match']) + df['distance']).mean())
    tonp(r, m_names)
    tonp(r, ['metric2'])
    return r

def readResults2(n, shortname, name):
    m_names = ['metric', 'match', 'distance', 'numf', 'diff']
    prefix = DATA_DIR + "/results/" + shortname + str(n) + "/result_" + name
    suffix = "_" + str(n) + "_10.csv.gz"
    r = {'metric':[], 'match' : [], 'distance' : [], 'numf' : [], 'diff' : [], 'metric2' : [] }
    df_full = pd.DataFrame()
    for i in range(1, 7) :
        fullName = prefix + str(i) + suffix
        df = pd.read_csv(fullName, compression='gzip')#.drop(['Unnamed: 0'],axis=1)
        df_full = df_full.append(df)
        add(r, m_names, df)
        r['metric2'].append(((1.0 / df['match']) + df['distance']).mean())
    tonp(r, m_names)
    tonp(r, ['metric2'])
    return df_full
```




```python
t2 = readResults(2, "t", "train")
t10 = readResults(10, "t", "train")
t50 = readResults(50, "t", "train")
t100 = readResults(100, "t", "train")
```




```python
v2 = readResults(2, "v", "validate")
v10 = readResults(10, "v", "validate")
v50 = readResults(50, "v", "validate")
v100 = readResults(100, "v", "validate")
```




```python
t2df = readResults2(2, "t", "train")
t10df = readResults2(10, "t", "train")
t50df = readResults2(50, "t", "train")
t100df = readResults2(100, "t", "train")
```




```python
t2df.agg([np.mean, np.std])
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
      <th>playlist_id</th>
      <th>n_clusters</th>
      <th>start_num</th>
      <th>metric</th>
      <th>match</th>
      <th>distance</th>
      <th>numf</th>
      <th>diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>492621.840113</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>41.498213</td>
      <td>6.019840</td>
      <td>23.742583</td>
      <td>6.903134</td>
      <td>17.367623</td>
    </tr>
    <tr>
      <th>std</th>
      <td>284337.026741</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>719.542719</td>
      <td>5.571262</td>
      <td>27.351196</td>
      <td>1.296133</td>
      <td>719.121609</td>
    </tr>
  </tbody>
</table>
</div>





```python
fig, ax = plt.subplots(1,4,figsize=(20,5))
t2df.drop(['playlist_id', 'n_clusters', 'start_num'], axis=1).boxplot(ax=ax[0])
ax[0].set_title('Metric Variability of 2 Cluster Model')
t10df.drop(['playlist_id', 'n_clusters', 'start_num'], axis=1).boxplot(ax=ax[1])
ax[1].set_title('Metric Variability of 10 Cluster Model')
t50df.drop(['playlist_id', 'n_clusters', 'start_num'], axis=1).boxplot(ax=ax[2])
ax[2].set_title('Metric Variability of 50 Cluster Model')
t100df.drop(['playlist_id', 'n_clusters', 'start_num'], axis=1).boxplot(ax=ax[3])
ax[3].set_title('Metric Variability of 100 Cluster Model')
fig.suptitle('Comparison of Metric Variability for Each Model')
fig.tight_layout(rect=[0, 0, 1, .95])
plt.show()
```



![png](display_resultsv2_files/display_resultsv2_7_0.png)


The above plots show that there is significant variability in each metric, this is most pronounced in the difference in predicted followers vs actual followers metric.



```python
fig, ax = plt.subplots(1,4,figsize=(20,5))
t2df.drop(['playlist_id', 'n_clusters', 'start_num', 'metric', 'diff'], axis=1).boxplot(ax=ax[0])
ax[0].set_title('Metric Variability of 2 Cluster Model')
t10df.drop(['playlist_id', 'n_clusters', 'start_num', 'metric', 'diff'], axis=1).boxplot(ax=ax[1])
ax[1].set_title('Metric Variability of 10 Cluster Model')
t50df.drop(['playlist_id', 'n_clusters', 'start_num', 'metric', 'diff'], axis=1).boxplot(ax=ax[2])
ax[2].set_title('Metric Variability of 50 Cluster Model')
t100df.drop(['playlist_id', 'n_clusters', 'start_num', 'metric', 'diff'], axis=1).boxplot(ax=ax[3])
ax[3].set_title('Metric Variability of 100 Cluster Model')
fig.suptitle('Comparison of Metric Variability for Each Model')
fig.tight_layout(rect=[0, 0, 1, .95])
plt.show()
```



![png](display_resultsv2_files/display_resultsv2_9_0.png)


The distance between playlists also has significant variability, one way we could possibly reduce this is to normalize the distance variables so that distance is a number between 0 and 1. 



```python
fig, ax = plt.subplots(1,4,figsize=(20,5))
ax[0].boxplot(t2df.match)
ax[0].set_title('Metric Variability of 2 Cluster Model')
ax[1].boxplot(t10df.match)
ax[1].set_title('Metric Variability of 10 Cluster Model')
ax[2].boxplot(t50df.match)
ax[2].set_title('Metric Variability of 50 Cluster Model')
ax[3].boxplot(t100df.match)
ax[3].set_title('Metric Variability of 100 Cluster Model')
for axis in ax:
    axis.set_xticklabels('')
fig.suptitle('Comparison of Metric Variability for Each Model')
fig.tight_layout(rect=[0, 0, 1, .95])
plt.show()
```



![png](display_resultsv2_files/display_resultsv2_11_0.png)




```python
fig, ax = plt.subplots(1,4,figsize=(20,5))
ax[0].boxplot(t2df.numf)
ax[0].set_title('Predicted Followers Variability of 2 Cluster Model')
ax[1].boxplot(t10df.numf)
ax[1].set_title('Predicted Followers Variability of 10 Cluster Model')
ax[2].boxplot(t50df.numf)
ax[2].set_title('Predicted Followers Variability of 50 Cluster Model')
ax[3].boxplot(t100df.numf)
ax[3].set_title('Predicted Followers Variability of 100 Cluster Model')
for axis in ax:
    axis.set_xticklabels('')
fig.suptitle('Comparison of Predicted Followers Variability for Each Model')
fig.tight_layout(rect=[0, 0, 1, .95])
plt.show()
```



![png](display_resultsv2_files/display_resultsv2_12_0.png)




```python
test2 = readResults(2, 'test', 'test')
test10 = readResults(10, 'test', 'test')
```




```python
def make_mean_metrics(models, sn):
    results_dict = {}
    #models = [t2, t10, t50, t100]
    names = [2, 10, 50, 100]
    for model, name in zip(models, names):
        model_params = {}
        for key in model:
            model_params[key] = np.mean(model[key])
            model_params[key+'std'] = np.std(model[key])
        results_dict[sn+str(name)] = model_params
    return results_dict
```




```python
models_train = [t2, t10, t50, t100]
train_dict = make_mean_metrics(models_train, 't')
results_train = pd.DataFrame.from_dict(train_dict).T
results_train
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
      <th>diff</th>
      <th>diffstd</th>
      <th>distance</th>
      <th>distancestd</th>
      <th>match</th>
      <th>matchstd</th>
      <th>metric</th>
      <th>metric2</th>
      <th>metric2std</th>
      <th>metricstd</th>
      <th>numf</th>
      <th>numfstd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>t2</th>
      <td>17.366214</td>
      <td>19.586043</td>
      <td>23.742464</td>
      <td>1.204582</td>
      <td>6.019813</td>
      <td>0.117456</td>
      <td>41.496689</td>
      <td>24.130509</td>
      <td>1.202147</td>
      <td>18.936747</td>
      <td>6.903080</td>
      <td>0.654374</td>
    </tr>
    <tr>
      <th>t10</th>
      <td>17.447881</td>
      <td>19.232502</td>
      <td>24.761268</td>
      <td>0.929312</td>
      <td>6.129652</td>
      <td>0.176668</td>
      <td>42.599327</td>
      <td>25.151356</td>
      <td>0.934653</td>
      <td>18.970186</td>
      <td>7.006165</td>
      <td>0.197402</td>
    </tr>
    <tr>
      <th>t50</th>
      <td>17.313173</td>
      <td>19.268623</td>
      <td>25.829698</td>
      <td>0.902331</td>
      <td>6.151795</td>
      <td>0.124508</td>
      <td>43.530253</td>
      <td>26.217108</td>
      <td>0.903429</td>
      <td>19.148228</td>
      <td>6.777501</td>
      <td>0.138788</td>
    </tr>
    <tr>
      <th>t100</th>
      <td>17.419482</td>
      <td>19.445556</td>
      <td>25.663053</td>
      <td>0.693858</td>
      <td>6.215457</td>
      <td>0.099706</td>
      <td>43.468382</td>
      <td>26.048898</td>
      <td>0.689577</td>
      <td>19.784372</td>
      <td>6.772136</td>
      <td>0.073181</td>
    </tr>
  </tbody>
</table>
</div>





```python
models_validate = [v2, v10, v50, v100]
validate_dict = make_mean_metrics(models_validate, 'v')
results_validate = pd.DataFrame.from_dict(validate_dict).T
results_validate
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
      <th>diff</th>
      <th>diffstd</th>
      <th>distance</th>
      <th>distancestd</th>
      <th>match</th>
      <th>matchstd</th>
      <th>metric</th>
      <th>metric2</th>
      <th>metric2std</th>
      <th>metricstd</th>
      <th>numf</th>
      <th>numfstd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>v2</th>
      <td>8.383877</td>
      <td>3.004734</td>
      <td>23.266120</td>
      <td>0.923800</td>
      <td>6.221482</td>
      <td>0.087430</td>
      <td>32.019275</td>
      <td>23.635325</td>
      <td>0.926099</td>
      <td>3.192091</td>
      <td>6.924442</td>
      <td>0.808971</td>
    </tr>
    <tr>
      <th>v10</th>
      <td>7.866056</td>
      <td>2.634214</td>
      <td>24.467220</td>
      <td>0.964562</td>
      <td>6.322889</td>
      <td>0.081322</td>
      <td>32.699332</td>
      <td>24.833359</td>
      <td>0.962880</td>
      <td>3.407503</td>
      <td>6.181138</td>
      <td>0.223869</td>
    </tr>
    <tr>
      <th>v50</th>
      <td>7.961983</td>
      <td>2.822155</td>
      <td>24.610007</td>
      <td>0.909844</td>
      <td>6.493108</td>
      <td>0.070977</td>
      <td>32.932138</td>
      <td>24.970164</td>
      <td>0.908917</td>
      <td>3.431358</td>
      <td>6.253282</td>
      <td>0.109693</td>
    </tr>
    <tr>
      <th>v100</th>
      <td>7.884393</td>
      <td>2.696971</td>
      <td>24.743925</td>
      <td>0.742098</td>
      <td>6.541668</td>
      <td>0.095974</td>
      <td>32.985807</td>
      <td>25.101392</td>
      <td>0.744146</td>
      <td>3.234505</td>
      <td>6.126036</td>
      <td>0.171505</td>
    </tr>
  </tbody>
</table>
</div>





```python
models_test = [test2, test10]
test_dict = make_mean_metrics(models_test, 'test')
results_test = pd.DataFrame.from_dict(test_dict).T
results_test
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
      <th>diff</th>
      <th>diffstd</th>
      <th>distance</th>
      <th>distancestd</th>
      <th>match</th>
      <th>matchstd</th>
      <th>metric</th>
      <th>metric2</th>
      <th>metric2std</th>
      <th>metricstd</th>
      <th>numf</th>
      <th>numfstd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test2</th>
      <td>13.347084</td>
      <td>7.996984</td>
      <td>25.105574</td>
      <td>1.956354</td>
      <td>6.189874</td>
      <td>0.139431</td>
      <td>38.832057</td>
      <td>25.48498</td>
      <td>1.956773</td>
      <td>8.834247</td>
      <td>7.429872</td>
      <td>0.323859</td>
    </tr>
    <tr>
      <th>test10</th>
      <td>12.797532</td>
      <td>8.089814</td>
      <td>24.755806</td>
      <td>0.846285</td>
      <td>6.294219</td>
      <td>0.155728</td>
      <td>37.930501</td>
      <td>25.13291</td>
      <td>0.847011</td>
      <td>8.529533</td>
      <td>6.795775</td>
      <td>0.237814</td>
    </tr>
  </tbody>
</table>
</div>





```python
plot_alpha = .1
fig, ax = plt.subplots(6,1, figsize=(10,20))
labels_train = ['Train Playlist Distance', 'Train Playlist Match Count', 'Train Combined Metric', 'Train Folowers', 
                'Train Follower Difference', 'Train Matching Songs plus Distance']
labels_validate = ['Validate Playlist Distance', 'Validate Playlist Match Count', 'Validate Combined Metric', 
                   'Validate Folowers', 'Validate Follower Difference', 'Validate Matching Songs plus Distance']
plot_order = ['distance', 'match', 'metric', 'numf', 'diff', 'metric2']
titles = ['Playlist Distance', 'Number of Matching Songs', 'Combined Metric', 'Predicted Number of Followers', 
          'Number of Follower Delta', 'Inverted Matching Songs plus Distance']
names = [2, 10, 50, 100]
# results_train.plot(y='diff', ax=ax[0])
# results_train.plot(y='distance', ax=ax[1])
# results_train.plot(y='match', ax=ax[2])
# results_train.plot(y='metric', ax=ax[3])
for axis, po, labelt, labelv, title in \
zip(ax, plot_order, labels_train, labels_validate, titles):
    results_train.plot(y=po, ax=axis, label=labelt)
    axis.fill_between(np.arange(4), results_train[po] + 2*results_train[po+'std'],
                      results_train[po] - 2*results_train[po+'std'], alpha=plot_alpha)
    results_validate.plot(y=po, ax=axis, label=labelv)
    axis.fill_between(np.arange(4), results_validate[po] + 2*results_validate[po+'std'],
                      results_validate[po] - 2*results_validate[po+'std'], alpha=plot_alpha)
    axis.set_xlabel('Cluster Size')
    axis.set_ylabel('Metric Score')
    axis.set_xticks(np.arange(4))
    axis.set_xticklabels(names)
    axis.set_title(title)
fig.tight_layout()
plt.show()
```



![png](display_resultsv2_files/display_resultsv2_18_0.png)




```python
results_validate_test = results_validate.loc[['v2', 'v10'], :]
results_train_test = results_train.loc[['t2', 't10'], :]
```




```python
po, results_train[po+'std']
```





    ('metric2', t2      1.202147
     t10     0.934653
     t50     0.903429
     t100    0.689577
     Name: metric2std, dtype: float64)





```python
fig, ax = plt.subplots(6,1, figsize=(10,20))
labels_train = ['Train Playlist Distance', 'Train Playlist Match Count', 'Train Combined Metric', 'Train Folowers', 
                'Train Follower Difference', 'Train Matching Songs plus Distance']
labels_validate = ['Validate Playlist Distance', 'Validate Playlist Match Count', 'Validate Combined Metric', 
                   'Validate Folowers', 'Validate Follower Difference', 'Validate Matching Songs plus Distance']
labels_test = ['Test Playlist Distance', 'Test Playlist Match Count', 'Test Combined Metric', 'Test Folowers', 
               'Test Follower Difference', 'Test Matching Songs plus Distance']
plot_order = ['distance', 'match', 'metric', 'numf', 'diff', 'metric2']
titles = ['Playlist Distance', 'Number of Matching Songs', 'Combined Metric', 'Predicted Number of Followers', 
          'Number of Follower Delta', 'Inverted Matching Songs plus Distance']
names = [2, 10]
# results_train.plot(y='diff', ax=ax[0])
# results_train.plot(y='distance', ax=ax[1])
# results_train.plot(y='match', ax=ax[2])
# results_train.plot(y='metric', ax=ax[3])
for axis, po, labelt, labelv, title, labeltest in \
zip(ax, plot_order, labels_train, labels_validate, titles, labels_test):
    results_train_test.plot(y=po, ax=axis, label=labelt)
    axis.fill_between(np.arange(2), results_train_test[po] + 2*results_train_test[po+'std'],
                      results_train_test[po] - 2*results_train_test[po+'std'], alpha=plot_alpha)
    results_validate_test.plot(y=po, ax=axis, label=labelv)
    axis.fill_between(np.arange(2), results_validate_test[po] + 2*results_validate_test[po+'std'],
                      results_validate_test[po] - 2*results_validate_test[po+'std'], alpha=plot_alpha)
    results_test.plot(y=po, ax=axis, label=labeltest)
    axis.fill_between(np.arange(2), results_test[po] + 2*results_test[po+'std'],
                      results_test[po] - 2*results_test[po+'std'], alpha=plot_alpha)
    axis.set_xlabel('Cluster Size')
    axis.set_ylabel('Metric Score')
    axis.set_xticks(np.arange(2))
    axis.set_xticklabels(names)
    axis.set_title(title)
fig.tight_layout()
plt.show()
```



![png](display_resultsv2_files/display_resultsv2_21_0.png)




```python
x = range(1, 7)
fig, axes = plt.subplots(1,2,figsize=(15,5))
axes[0].set_title('Plot of Loss - Train')
axes[0].plot(x, t2['metric'], label='2 Clusters')
axes[0].plot(x, t10['metric'], label='10 Clusters')
axes[0].plot(x, t50['metric'], label='50 Clusters')
axes[0].plot(x, t100['metric'], label='100 Clusters')
axes[0].set_xlabel('Batch Number')
axes[0].set_ylabel('Loss Metric')
axes[0].legend()
axes[1].set_title('Plot of Loss - Validate')
axes[1].plot(x, v2['metric'], label='2 Clusters')
axes[1].plot(x, v10['metric'], label='10 Clusters')
axes[1].plot(x, v50['metric'], label='50 Clusters')
axes[1].plot(x, v100['metric'], label='100 Clusters')
axes[1].set_xlabel('Batch Number')
axes[1].set_ylabel('Loss Metric')
axes[1].legend()
plt.show()
```



![png](display_resultsv2_files/display_resultsv2_22_0.png)


Looking at the combined metric by batch shows that each model reacted to the individual batches in a fairly consistent manner. This is slightly less true for the validation set, but the relationship is still there for at least the 10, 50 and 100 cluster models.



```python
x = range(1, 7)
fig, axes = plt.subplots(3,1,figsize=(15,15))
axes = axes.ravel()
axes[0].set_title('Plot of Loss (Song Matches plus Distance) - Train')
axes[0].plot(x, t2['metric2'], label='2 Clusters')
axes[0].plot(x, t10['metric2'], label='10 Clusters')
axes[0].plot(x, t50['metric2'], label='50 Clusters')
axes[0].plot(x, t100['metric2'], label='100 Clusters')
axes[0].set_xlabel('Batch Number')
axes[0].set_ylabel('Loss Metric')
axes[0].legend()
axes[1].set_title('Plot of Loss (Song Matches plus Distance) - Validate')
axes[1].plot(x, v2['metric2'], label='2 Clusters')
axes[1].plot(x, v10['metric2'], label='10 Clusters')
axes[1].plot(x, v50['metric2'], label='50 Clusters')
axes[1].plot(x, v100['metric2'], label='100 Clusters')
axes[1].set_xlabel('Batch Number')
axes[1].set_ylabel('Loss Metric')
axes[1].legend()
axes[2].set_title('Plot of Loss (Song Matches plus Distance) - Test')
axes[2].plot(x, test2['metric2'], alpha=0.5, color='b', label='2 clusters')
axes[2].plot(x, test10['metric2'], alpha=0.5, color='r', label='10 clusters')
axes[2].set_xlabel('batch')
axes[2].set_ylabel('loss metric')
axes[2].legend()
plt.show()
```



![png](display_resultsv2_files/display_resultsv2_24_0.png)


We created a second combined metric aimed at reducing the variability, this metric has results that are more in line with expectations with the model performing slightly better on the train set than the other sets. 
