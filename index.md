---
title: Spotify Playlist Generation
---

The purpose of this project is to explore methods learned in the CS109A Data Science Course to create a model which can generate Spotify playlists.

The project has two goals:

**Goal 1: automatic playlist generation**

Would using playlist engineered features based on songs allow a model creation that classifies the "base" playlist as “close” to some known playlists? Can number of followers be estimated from the rest of the playlist features?

**Goal 2: cold start**

Can similarity be used to add some (how many?) songs to "base" playlist, and proceed with algorithm written for goal 1?

Assumption made for now is that “base” playlist is most important, context information can be used for tie braking and sorting (e.g “studying” – less loud songs)
