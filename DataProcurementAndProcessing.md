---
title: DataProcurementAndProcessing
nav_include: 3
---

#### Data procurement

To get data that describes songs ("audio features" in Spotify), we queried Spotify API for song identifier (Spotify id).
We tried to fetch song similarity data from Spotify but that was taking too much time and hence wasn't added to our dataset.
Overall, we used compressed csv files to store intermediate and final data sets.

#### Data processing

* All names (playlists names, song names) where cleaned by removing punctuation, extra spaces, etc
* Songs in playlists processed to find co-occurences of songs in playlists. Our take is that song co-occurence in playlist shows signal for song similarity (of course, Spotify promoted songs and "hits" add noise). If songs are found in the same list - that adds a count and this pair is saved into resulting set.

Sources below. These are Python scripts, not Jupyter notebooks to decrease memory pressure and let scripts run unattended.



```python
# Preprocessing:
#!/usr/bin/env python
# coding: utf-8

import sys
import json
import codecs
import datetime
import numpy as np
import pandas as pd
import re
import time
import gzip
import csv

DATA_DIR = "./data/data/"
pretty = True
compact = False
cache = {}

def cleanString(s):
    s = re.sub(r'[^\w\s]','', s) # remove punctuation
    s = re.sub(' +', ' ', s) # remove double spaces
    s = "".join(i for i in s if ord(i)<128) # remove non-ascii letters
    return s.lower().strip()

def getId(dict, str) :
    s = cleanString(str)
    if len(s) == 0 :
        return -1
    if s in dict :
        id = dict[s]
    else :
        id = len(dict)
        dict[s] = id
    return id

def getSongId(dict, track, s_id) :
    s = cleanString(track['track_name'])
    if len(s) == 0 :
        return (-1, s_id)
    if s in dict :
        return (dict[s][0], s_id)
    id = s_id
    s_id += 1
    dict[s] = [ id, track['track_uri'].split(':')[2], float(track['duration_ms']) / 1000 ]
    return (id, s_id)

def getPlaylId(dict, playlist) :
    s = cleanString(playlist['name'])
    if len(s) == 0 :
        return -1
    id = len(dict)
    dict[id] = [ s, int(playlist['num_followers']), 1 if bool(playlist['collaborative']) else 0,
                int(playlist['num_tracks']), int(playlist['num_albums']) ]
    return id

def full_playlist(start, end, showOnly=True, namesOnly=False, data=None, playlData=None, lastIndex=0):
    playlists = None
    prevFile = None
    playl = {}
    artists = {}
    albums = {}
    songs = {}
    p_id = 0
    s_id = 0
    for pid in range(int(start), int(end)):
        low = 1000 * int(pid / 1000)
        high = low + 999
        offset = pid - low
        path = DATA_DIR + "/mpd.slice." + str(low) + '-' + str(high) + ".json"
        if prevFile is None or prevFile != path:
            f = codecs.open(path, 'r', 'utf-8')
            js = f.read()
            f.close()
            playlists = json.loads(js)
            prevFile = path
            print("File", path)

        playlist = playlists['playlists'][offset]  # ??
        if showOnly:
            print_playlist(playlist)
        else:
            playlistId = getPlaylId(playl, playlist)
            if playlistId == -1 :
                # skip this playlist
                continue

            if namesOnly:
                playlData[pid][0] = pid
                playlData[pid][1] = cleanString(playlist['name'])
            else:
                for track in playlist['tracks']:
                    artistId = getId(artists, track['artist_name'])
                    albumId = getId(albums, track['album_name'])
                    (songId, s_id) = getSongId(songs, track, s_id)
                    if artistId == -1 or albumId == -1 or songId == -1 :
                        continue

                    data[str(playlist['pid']) + "-" + str(track['pos'])] = [ p_id, songId, albumId, artistId, playlistId ]
                    p_id += 1
    return (playl, artists, albums, songs)

def do_playlists_in_range(start, end, showOnly=True, namesOnly=False, data=None, playlData=None):
    try:
        istart = int(start)
        iend = int(end)
        if istart <= iend and istart >= 0 and iend <= 1000000:
            i = 0
            for pid in range(istart, iend):
                if showOnly:
                    do_playlist(pid)
                else:
                    i = do_playlist(pid, showOnly=False, namesOnly=namesOnly, data=data, playlData=playlData,
                                    lastIndex=i)
            return i
    except:
        raise
        print("bad pid")

import csv

def writeToCsv(dict, fname, col = None) :
    columns = ['name', 'id'] if col is None else col
    with gzip.open(DATA_DIR + fname, 'wt', newline='') as fz:
        writer = csv.writer(fz, delimiter=',')
        writer.writerow(columns)
        for (k, v) in dict.items() :
            p = list()
            p.append(k)
            if col == None :
                p.append(v)
            else:
                p.extend(v)
            writer.writerow(p)

data = {}
colsTracks = ['pidpos', 'id', 'track_id', 'album_id', 'artist_id', 'playlist_id']

start = time.time()
(playl, artists, albums, songs) = full_playlist(0, 999999, showOnly=False, namesOnly=False, data=data, playlData=None)
print("calced", (time.time() - start), "from start")
writeToCsv(playl, 'playlists.csv.gz', ['id', 'name', 'num_followers', 'collaborative', 'num_tracks', 'num_albums'])
writeToCsv(artists, 'artists.csv.gz')
writeToCsv(albums, 'albums.csv.gz')
writeToCsv(songs, 'songs.csv.gz', ['name', 'id', 'uri', 'duration_ms'])
writeToCsv(data, 'preproc.csv.gz', colsTracks)
print("Done", (time.time() - start), "from start")
```




```python
# Song info fetch. Several files were stored and merged to account for transmission failures, etc.
import urllib
import urllib.request as request
import json
import time
import gzip
import csv
import sys

regSleep = 1.0 / 165;
authH = {'Accept': 'application/json',
    'User-agent': 'Mozilla/5.0',
    "Content-Type": "application/json",
    'Authorization' : 'Bearer {}'.format(
    sys.argv[3]
    ) }

startId = sys.argv[2]
searchFirst = True
DATA_DIR = "./data/data/"
with gzip.open(DATA_DIR + sys.argv[1] + "_aug_songs_" + (startId if startId is not None else "") + ".csv.gz", 'wt') as fz:
    writer = csv.writer(fz, delimiter=',')
    if startId is None :
        writer.writerow(['name', 'id', 'uri', 'duration', 'danceability', 'energy', 'key', 'loudness', 'mode',
               'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'])
    with gzip.open(DATA_DIR + "songs.csv.gz", mode="rt") as f:
        print("file", f)
        csvobj = csv.reader(f, delimiter=',', quotechar="'")
        first = True
        fullStop = False
        for line in csvobj:
            if first :
                first = False
                continue
            if len(line) == 0 :
                continue
            id = line[2]
            print("->", id)
            if startId is not None and searchFirst:
               if startId == id :
                   print("found", startId)
                   searchFirst = False
               else:
                   continue
            for retry in range(0, 10):
                code = -1
                try :
                    req = request.Request(url = "https://api.spotify.com/v1/audio-features/{}".format(id), headers=authH)
                    resp = request.urlopen(req)
                    content = resp.read()
                    resp = json.loads(content)
                    out = [line[0], line[1], line[2], line[3], resp['danceability'], resp['energy'], resp['key'], resp['loudness'], resp['mode'],
                           resp['speechiness'], resp['acousticness'], resp['instrumentalness'], resp['liveness'], resp['valence'], resp['tempo'],
                           resp['time_signature'] ]
                    writer.writerow(out)
                    time.sleep(regSleep)
                    break
                except urllib.error.HTTPError  as e:
                    code = e.code
                    print("HttpCode", code)
                except urllib.error.URLError as e:
                    if hasattr(e, 'reason'):
                        print("url error - reason", e.reason)
                        print("retry")
                        time.sleep(5)
                        continue
                    elif hasattr(e, 'code'):
                        code = e.code
                        print("url error code", code)
                if code == 404 :
                    print("not found - skipping")
                    break
                if code == 429 :
                    print("rate limit")
                if code == 401 :
                    print("need token")
                    fullStop = True
                    break
                print("sleeping")
                time.sleep(10)
            if fullStop :
                break # exit
print("-----")
```




```python
# Song similarity - done in chunks to fit into memory
import gzip
import csv
import sys
import numpy as np

def calcPlayl(plId, songs) :
    n = len(songs)
    if n == 0 or n == 1:
        return
    print (plId, "len", n)
    r = range(0, n)
    for i in r :
        if matrix[songs[i], 0] is None :
            matrix[songs[i], 0] = {}
        for j in range(i + 1, n) :
            otherId = songs[j]
            if otherId in matrix[songs[i], 0] :
                matrix[songs[i], 0][otherId] += 1
            else :
                matrix[songs[i], 0][otherId] = 1


DATA_DIR = "./data/data/"
numSongs = 1389689 + 1 # max song id , starts from 0
matrix = np.empty([numSongs, 1], dtype=np.dtype('O'))

cnt = 0
MAX = 233000
runId = int(sys.argv[1])
startPl = runId * MAX
endPl = startPl + MAX
with gzip.open(DATA_DIR + "preproc.csv.gz", mode="rt") as f:
    print("file", f)
    csvobj = csv.reader(f, delimiter=',', quotechar="'")
    first = True
    prevPl = -1
    songs = []
    for line in csvobj:
        if first :
            first = False
            continue
        if len(line) == 0 :
            continue
        songId = int(line[2])
        plId = int(line[5])
        if plId < startPl :
            continue
        if plId > endPl :
            break
        if plId != prevPl :
            calcPlayl(prevPl, songs)
            songs = []
        songs.append(songId)
        prevPl = plId

print("done from ", startPl, "to", endPl)
with gzip.open(DATA_DIR + str(runId) + "_simsong_calc_.csv.gz", 'wt') as fz:
    writer = csv.writer(fz, delimiter=',')
    writer.writerow(['songid', 'simsongid', 'count'])
    for i in range(0, matrix.shape[0]) :
        if matrix[i, 0] is None :
            continue
        for key, value in sorted(matrix[i, 0].items(), key=lambda kv: kv[1], reverse=True) :
            writer.writerow([i, key, value])

```




```python
# Song similarity merge
import gzip
import csv
import sys

def empty(line1) :
    return line1 is None or len(line1) == 0

def allEmpty(dones) :
    for done in dones :
       if not done:
           return False
    return True

def writeSims(writer, songId, simSongs):
    for key, value in sorted(simSongs.items(), key=lambda kv: kv[1], reverse=True) :
        # cutoff ?
        writer.writerow([songId, key, value])

#return false if song ids don't match
def addSongInfo(songId, line, simSongs):
    # print("add", songId, line, simSongs)
    newSongId = int(line[0])
    if songId == newSongId :
        simId = int(line[1])
        cnt = int(line[2])
        if simId in simSongs :
            simSongs[simId] += cnt
        else :
            simSongs[simId] = cnt
        return True
    return False

#return empty list if end of file is reached, next-song line otherwise
def fetchSongInfo(songId, prevLine, curIter, simSongs) :
    # print("fetch", songId, prevLine)
    if not empty(prevLine) :
        match = addSongInfo(songId, prevLine, simSongs)
        if not match :
            return prevLine
    while True :
        line = next(curIter, None)
        if empty(line):
            return []
        match = addSongInfo(songId, line, simSongs)
        if not match :
            return line

DATA_DIR = "./data/data/"

numSongs = 1389689 + 1 # max song id , starts from 0
numFiles = 2
files = [None]*numFiles
csvobj = [None]*numFiles
iters = [None]*numFiles
prevLine  = [None]*numFiles
doneFile = [False]*numFiles
simSongs = {}
fCnt = 0
file0 = sys.argv[1]
file1 = sys.argv[2]
with gzip.open(DATA_DIR + file0 + "_" + file1 + "_simsong_calc_.csv.gz", 'wt') as fz:
    with gzip.open(DATA_DIR + file0 + "_simsong_calc_.csv.gz", mode="rt") as files[fCnt]:
        csvobj[fCnt] = csv.reader(files[fCnt], delimiter=',', quotechar="'")
        iters[fCnt] = iter(csvobj[fCnt])
        next(iters[fCnt])
        fCnt += 1
        with gzip.open(DATA_DIR + file1 + "_simsong_calc_.csv.gz", mode="rt") as files[fCnt]:
            csvobj[fCnt] = csv.reader(files[fCnt], delimiter=',', quotechar="'")
            iters[fCnt] = iter(csvobj[fCnt])
            next(iters[fCnt])
            writer = csv.writer(fz, delimiter=',')
            writer.writerow(['songid', 'simsongid', 'count'])
            print("files", files, "to", fz)
            for songId in range(0, numSongs) :
                # print("song", songId)
                for i in range(0, numFiles) :
                    if not doneFile[i] :
                        # print("file", i)
                        prevLine[i] = fetchSongInfo(songId, prevLine[i], iters[i], simSongs)
                        doneFile[i] = empty(prevLine[i])
                        # print("after proc prevLine", prevLine[i])
                writeSims(writer, songId, simSongs)
                simSongs = {}
                if allEmpty(doneFile) :
                    # full stop
                    break

print("done ")
```

