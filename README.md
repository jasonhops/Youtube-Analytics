# Youtube-Analytics
The purpose of this project is to demonstrate our ability to apply analytics to a new YouTube channel for growing the number of views and subscribers.

Script 1) Python: pull uploads + engineer ~50 features (Data API)

This script:
- Resolves channel ID from the handle using search.list(type='channel', q='@spaceMSDA') (no direct “handle → channel” endpoint)  ￼
- Gets the uploads playlist from channels.list(part=contentDetails) and paginates playlistItems.list to collect all video IDs.  ￼
- Calls videos.list(part=snippet,statistics,contentDetails) in batches to obtain title/desc/tags/duration/… and builds feature columns.  ￼
