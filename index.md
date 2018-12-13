---
title: Spotify Playlist Generation
---

The purpose of this project is to explore methods learned in the CS109A Data Science Course to create a model which can generate Spotify playlists.

Introduction: 

**Spotify** 

Spotify is a music service that offers audio playback over the Internet. It offers a developer access (https://developer.spotify.com) that allows querying of data and manipulation of user data. Users benefit from listening to the music and create playlists which can be followed by other users. Spotify provides access to some of the datasets and is interested in improvement of auto-generation of playlists.

The project has two goals:

**Goal 1: automatic playlist generation**

Would using playlist engineered features based on songs allow a model creation that classifies the "base" playlist as “close” to some known playlists? Can number of followers be estimated from the rest of the playlist features?

**Goal 2: cold start**

Can similarity be used to add songs to "base" playlist, and proceed with algorithm written for goal 1?

Assumption made is that “base” playlist is most important and song selection can be made by utilizing "similar" songs and playlists. In the model created for this projects, these goals are combined. I.e. using a "nucleus" song, we first (recursively) add similar songs (calculated and stored ahead of time for all known songs) and then add songs from "close" ("similar") playlists. Playlist closeness is calculated through KMeans clustering on engineered features which are aggregation of songs' features in the playlist.
