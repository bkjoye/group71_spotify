#!/usr/bin/env python
# coding: utf-8

# In[20]:


import sys
import datetime
import numpy as np
import pandas as pd
import string
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import gzip
import csv
from multiprocessing import Process
from sklearn.utils import shuffle

DATA_DIR="./data/data"


# In[2]:


df = pd.read_csv(DATA_DIR + '/pidpos.csv.gz', compression='gzip').drop(['Unnamed: 0'],axis=1)
dfAugSongs = pd.read_csv(DATA_DIR + '/full_aug_songs.csv.gz', compression='gzip')
dfPlaylists = pd.read_csv(DATA_DIR + '/playlists.csv.gz', compression='gzip')
# dfPlSongsAgg = pd.read_csv(DATA_DIR + '/raw_aug_playlists.csv.gz', compression='gzip')
dfTrain = pd.read_csv(DATA_DIR + '/train_aug_playlists.csv.gz', compression='gzip').drop(['Unnamed: 0'],axis=1)
dfValidate = pd.read_csv(DATA_DIR + '/validate_aug_playlists.csv.gz', compression='gzip').drop(['Unnamed: 0'],axis=1)
# dfTest = pd.read_csv(DATA_DIR + '/test_aug_playlists.csv.gz', compression='gzip').drop(['Unnamed: 0'],axis=1)
# read train / validate / test


# In[3]:


dfSim = pd.read_csv(DATA_DIR + '/simsong5.csv.gz', compression='gzip').drop(['Unnamed: 0'],axis=1)


# In[4]:


def addSim(dfSim, cur_set, c_id, num) :
    dfCandidates = dfSim[(dfSim.songid == c_id) | (dfSim.simsongid == c_id)]
    t = dfCandidates.sort_values(by='count', ascending=False).values[0:num, :]
    for i in t :
        id = i[1] if i[0] == c_id else i[0]
        if id in cur_set :
            continue
        cur_set.append(id)
    return cur_set


# In[5]:


def getPlAgg(candidate_pl) :
     return [ candidate_pl.danceability.mean(), candidate_pl.energy.mean(), 
                     candidate_pl.speechiness.mean(), candidate_pl.acousticness.mean(), 
                     candidate_pl.instrumentalness.mean(), candidate_pl.liveness.mean(),
                     candidate_pl.valence.mean(), candidate_pl.duration.mean(), candidate_pl.key.max(), 
                     candidate_pl.loudness.max(), 
                     candidate_pl.tempo.max(), candidate_pl.time_signature.max() ]# .iloc[0] -> first


# In[6]:


def getNumfXy(dfPlSongsAgg) :
    y = dfPlSongsAgg.num_followers
    X = dfPlSongsAgg[['mean_danceability','mean_energy','mean_speechiness','mean_acousticness',
                              'mean_instrumentalness', 'mean_liveness', 'mean_valence',
                 'mean_duration', 'max_key', 'max_loudness', 'max_tempo', 'max_time_signature']].values
    return (X, y)

def getNumfRegr(dfPlSongsAgg) :
    (X, y) = getNumfXy(dfPlSongsAgg)
    return LinearRegression().fit(X, y)


# In[7]:


def scoreNumfRegr(reg, dfValidate) :
    (X, y) = getNumfXy(dfValidate)
    return reg.score()

def getSongsFrom(dfSongMatch, n) :
    return dfSongMatch.sample(n, random_state=0)['track_id'].values.tolist()

def getSimSongs(dfSim, song_set, start_num, from_i, n) :
    size = len(song_set)
    while True :
        if size > n or from_i > (size - 1):
            break
        addSim(dfSim, song_set, song_set[from_i], start_num)
        from_i +=1
    song_set = song_set[0: n]
    assert len(set(song_set)) == len(song_set) # no dups expected
    return song_set
                    
cluster_columns = ['mean_danceability','mean_energy','mean_speechiness','mean_acousticness',
                          'mean_instrumentalness', 'mean_liveness', 'mean_valence',
             'mean_duration', 'max_key', 'max_loudness', 'max_tempo', 'max_time_signature']


# In[23]:


dfTrain.dropna(inplace=True)
reg = getNumfRegr(dfTrain) # training
# dfTrain=shuffle(dfTrain)


# In[24]:


dfValidate.dropna(inplace=True)
dfValidate = shuffle(dfValidate)
# dfTest.dropna(inplace=True)


# In[26]:


# start_num : how many similar songs to fetch on each level
def generate_playlists(dfPlSongsAgg, dfPidPosPl, name, reg, try_clusters, try_startNum) :
    num_top_pl = 5 # choose top 5 playlists
    train_song_id = 0 # first song is used to continue playlist
    toCluster = dfPlSongsAgg[cluster_columns].values
    for n_clusters in try_clusters :
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_jobs = -1).fit(toCluster)
        print("Found", n_clusters, "clusters")
        train_pl = dfPlSongsAgg.playlist_id.values
        for start_num in try_startNum :
            print("start_num", start_num)
            with gzip.open(DATA_DIR + "/result_" + name + "_" + str(n_clusters) + "_" + str(start_num) + ".csv.gz", 
                           'wt', newline='') as fz:
                writer = csv.writer(fz, delimiter=',')
                writer.writerow(['playlist_id', 'n_clusters', 'start_num', 'metric', 'match', 'distance', 'numf', 
                                 'diff'])
                for pl_id in train_pl :
                    print("Running playlist", pl_id)
                    try :
                        train_numf = dfPlSongsAgg[dfPlSongsAgg.playlist_id == pl_id].num_followers.values
                        target_n = int(dfPlSongsAgg[dfPlSongsAgg.playlist_id == pl_id].sum_num_tracks.values)
                        (train_agg_info, _) = getNumfXy(dfPlSongsAgg[dfPlSongsAgg.playlist_id == pl_id])
                        train_song_set = df[df.playlist_id == pl_id].track_id.values 
                        root_name = dfAugSongs[dfAugSongs.id == train_song_set[train_song_id]].name.values[0]
                        fromsim_n = int(target_n / 2) # metaparam
                        root_id = dfAugSongs[dfAugSongs.name == root_name].id.values[0]
                        song_set = [train_song_set[0]]
                        addSim(dfSim, song_set, root_id, start_num)
                        # choose start_num - loop to fromsim_n
                        getSimSongs(dfSim, song_set, start_num, 0, fromsim_n)
                        # find song aug data and create playlist
                        candidate_pl = dfAugSongs.iloc[song_set, :]
                        candidate_pl_agg = getPlAgg(candidate_pl)
                        p_label = kmeans.predict([candidate_pl_agg])[0]
                        dfPlaylistMatch = dfPlSongsAgg[kmeans.labels_ == p_label]
                        dfPlaylistMatchTop = dfPlaylistMatch.sort_values(by='num_followers', ascending=False).head(
                            num_top_pl)
                        dfSongMatch = pd.merge(dfPidPosPl, dfPlaylistMatchTop, left_on='playlist_id', right_on='playlist_id', 
                                               how='left').dropna()
                        from_cluster = target_n - fromsim_n
#                         print("fromsim_n", fromsim_n,"target_n",target_n)
                        if from_cluster > dfSongMatch.shape[0] : # not enough songs
                            song_from_pl = getSongsFrom(dfSongMatch, dfSongMatch.shape[0])
                            getSimSongs(dfSim, song_from_pl, start_num, 0, from_cluster)
                        else :    
                            song_from_pl = getSongsFrom(dfSongMatch, from_cluster)
                        song_play = [*song_set, *song_from_pl]
                        song_info = dfAugSongs.iloc[song_play, :]
                        # metrics
                        metric_match = list(set(train_song_set) & set(song_play))
                        metric_match_n = len(metric_match)
                        song_agg = np.array(getPlAgg(song_info)) # aggregated to playlist
                        dist = (song_agg - train_agg_info)**2
                        dist = np.sum(dist, axis=1)
                        dist = np.sqrt(dist)[0]
                        song_numf = round(reg.predict(song_agg.reshape(1, -1))[0])
                        numf_diff = int(np.abs(train_numf - song_numf)[0])
                        sum_metric = 1.0 / metric_match_n + dist + numf_diff
                        writer.writerow([pl_id, n_clusters, start_num, round(sum_metric, 2), metric_match_n, 
                                         round(dist, 2), song_numf, numf_diff])
                    except  Exception as e:
                        print("Ex playlist", pl_id, str(e))


# In[ ]:


kT = int(dfValidate.shape[0] / 6)
# limit = 500
n_cl = [2,10,50,100]
def func1() :
    print("starting 1")
    dfT = dfValidate.iloc[0 : kT, :]
    dfP = pd.merge(df, dfT, left_on='playlist_id', right_on='playlist_id', how='left').dropna()
    print("running 1")
    generate_playlists(dfT, dfP, "validate1", reg, try_clusters=n_cl, try_startNum=[10])
    print("done 1")
def func2() :
    print("starting 2")
    dfT = dfValidate.iloc[kT : 2*kT, :]
    dfP = pd.merge(df, dfT, left_on='playlist_id', right_on='playlist_id', how='left').dropna()
    print("running 2")
    generate_playlists(dfT, dfP, "validate2", reg, try_clusters=n_cl, try_startNum=[10])
    print("done 2")
def func3() :
    print("starting 3")
    dfT = dfValidate.iloc[2*kT : 3*kT, :]
    dfP = pd.merge(df, dfT, left_on='playlist_id', right_on='playlist_id', how='left').dropna()
    print("running 3")
    generate_playlists(dfT, dfP, "validate3", reg, try_clusters=n_cl, try_startNum=[10])
    print("done 3")
def func4() :
    print("starting 4")
    dfT = dfValidate.iloc[3*kT : 4*kT, :]
    dfP = pd.merge(df, dfT, left_on='playlist_id', right_on='playlist_id', how='left').dropna()
    print("running 4")
    generate_playlists(dfT, dfP, "validate4", reg, try_clusters=n_cl, try_startNum=[10])
    print("done 4")
def func5() :
    print("starting 5")
    dfT = dfValidate.iloc[4*kT : 5*kT, :]
    dfP = pd.merge(df, dfT, left_on='playlist_id', right_on='playlist_id', how='left').dropna()
    print("running 5")
    generate_playlists(dfT, dfP, "validate5", reg, try_clusters=n_cl, try_startNum=[10])
    print("done 5")
def func6() :
    print("starting 6")
    dfT = dfValidate.iloc[5*kT :, :]
    dfP = pd.merge(df, dfT, left_on='playlist_id', right_on='playlist_id', how='left').dropna()
    print("running 6")
    generate_playlists(dfT, dfP, "validate6", reg, try_clusters=n_cl, try_startNum=[10])
    print("done 6")


# In[ ]:


# continue_play('tonight tonight', 50)
def run() :
    p1 = Process(target=func1)
    p1.start()
    p2 = Process(target=func2)
    p2.start()
    p3 = Process(target=func3)
    p3.start()
    p4 = Process(target=func4)
    p4.start()
    p5 = Process(target=func5)
    p5.start()
    p6 = Process(target=func6)
    p6.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    print("Validation done")

run()
# generate_playlists(dfValidate.iloc[[1,2], :], df, "validate", reg, try_clusters=[2, 10], try_startNum=[10])
# print("Validate done")
# generate_playlists(dfTest.iloc[[2, 3], :], df, "test", reg, try_clusters=[2, 10], try_startNum=[10])
# print("Test done")


# In[ ]:




