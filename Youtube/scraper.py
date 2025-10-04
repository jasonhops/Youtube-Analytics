# pip install google-api-python-client pandas python-dateutil numpy
import os, math, re, json, time
from datetime import datetime, timezone
from dateutil.parser import isoparse
import numpy as np
import pandas as pd
from googleapiclient.discovery import build

API_KEY = os.getenv("YOUTUBE_API_KEY")  # set this in your environment
HANDLE  = "@spaceMSDA"

# ---------- Helpers ----------
def yt_build():
    return build("youtube", "v3", developerKey=API_KEY)

def get_channel_id_from_handle(youtube, handle):
    # Use search.list because there is no official handle->channelId method
    resp = youtube.search().list(
        q=handle, type="channel", part="snippet", maxResults=1
    ).execute()  # docs: search.list
    # pick first match
    items = resp.get("items", [])
    if not items:
        raise RuntimeError(f"Handle {handle} not found.")
    return items[0]["snippet"]["channelId"]

def get_uploads_playlist_id(youtube, channel_id):
    ch = youtube.channels().list(
        part="contentDetails", id=channel_id
    ).execute()  # docs: channels.list
    items = ch.get("items", [])
    if not items:
        raise RuntimeError("Channel not found.")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

def list_all_video_ids(youtube, uploads_playlist_id):
    ids = []
    token = None
    while True:
        pl = youtube.playlistItems().list(
            part="contentDetails", playlistId=uploads_playlist_id,
            maxResults=50, pageToken=token
        ).execute()  # docs: playlistItems.list
        ids += [it["contentDetails"]["videoId"] for it in pl.get("items", [])]
        token = pl.get("nextPageToken")
        if not token:
            break
    return ids

ISO_DUR_RE = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
def iso_duration_to_seconds(d):
    m = ISO_DUR_RE.match(d)
    h = int(m.group(1) or 0); mm = int(m.group(2) or 0); s = int(m.group(3) or 0)
    return h*3600 + mm*60 + s

def safe_int(x): 
    try: return int(x)
    except: return np.nan

def extract_thumb_metrics(snippet):
    thumbs = snippet.get("thumbnails", {})
    # choose highest available
    cand = []
    for k in ["maxres","standard","high","medium","default"]:
        if k in thumbs:
            t = thumbs[k]
            cand.append((t.get("width",0)*t.get("height",0), t))
    if not cand:
        return pd.Series(dict(th_w=np.nan, th_h=np.nan, th_area=np.nan))
    area, t = max(cand, key=lambda z:z[0])
    return pd.Series(dict(th_w=t.get("width"), th_h=t.get("height"), th_area=area))

def text_features(title, desc, tags):
    title = title or ""
    desc = desc or ""
    tags = tags or []
    return pd.Series({
        "title_len_chars": len(title),
        "title_len_words": len(title.split()),
        "desc_len_chars": len(desc),
        "desc_len_words": len(desc.split()),
        "tag_count": len(tags),
        "has_question": int("?" in title),
        "has_numbers": int(bool(re.search(r"\d", title))),
        "has_emoji": int(bool(re.search(r"[\U0001F300-\U0001FAFF]", title))),
        "caps_ratio": (sum(1 for c in title if c.isupper()) / max(1,len(title))),
        "kw_space": int(bool(re.search(r"\bspace|nebula|galaxy|jwst|hubble\b", (title+" "+desc).lower()))),
    })

def time_features(published_at):
    dt = isoparse(published_at).astimezone(timezone.utc)
    return pd.Series({
        "published_hour_utc": dt.hour,
        "published_dow": dt.weekday(),  # 0=Mon
        "published_month": dt.month,
        "age_days": (datetime.now(timezone.utc)-dt).days
    })

def ratio(x,y):
    return float(x)/float(y) if (x is not None and y not in (0,None)) else np.nan

# ---------- Main pull ----------
youtube = yt_build()
channel_id = get_channel_id_from_handle(youtube, HANDLE)  # search.list
uploads_pid = get_uploads_playlist_id(youtube, channel_id) # channels.list

video_ids = list_all_video_ids(youtube, uploads_pid)       # playlistItems.list
# batch videos.list calls (50 ids per request)
rows = []
for i in range(0, len(video_ids), 50):
    batch = video_ids[i:i+50]
    v = youtube.videos().list(
        part="snippet,statistics,contentDetails", id=",".join(batch)
    ).execute()  # docs: videos.list
    for it in v.get("items", []):
        sn = it["snippet"]; st = it.get("statistics", {}); cd = it.get("contentDetails", {})
        base = {
            "video_id": it["id"],
            "title": sn.get("title",""),
            "publishedAt": sn.get("publishedAt"),
            "categoryId": sn.get("categoryId"),
            "duration_sec": iso_duration_to_seconds(cd.get("duration","PT0S")),
            "definition": cd.get("definition"),
            "licensedContent": int(cd.get("licensedContent", False)),
            "viewCount": safe_int(st.get("viewCount")),
            "likeCount": safe_int(st.get("likeCount")),
            "commentCount": safe_int(st.get("commentCount")),
        }
        tf = text_features(sn.get("title"), sn.get("description"), sn.get("tags", []))
        th = extract_thumb_metrics(sn)
        tm = time_features(sn.get("publishedAt"))

        row = {**base, **tf.to_dict(), **th.to_dict(), **tm.to_dict()}
        # simple engagement features
        row.update({
            "likes_per_view": ratio(row["likeCount"], row["viewCount"]),
            "comments_per_view": ratio(row["commentCount"], row["viewCount"]),
            "views_per_day": ratio(row["viewCount"], max(1,row["age_days"])),
            "is_short": int(row["duration_sec"] <= 60),
            "is_mid": int(60 < row["duration_sec"] <= 600),
            "is_long": int(row["duration_sec"] > 600),
        })
        rows.append(row)

df = pd.DataFrame(rows)
df.sort_values("publishedAt", inplace=True)
df.to_csv("spaceMSDA_dataapi_features.csv", index=False)
print(f"Saved {len(df)} videos with {df.shape[1]} columns to spaceMSDA_dataapi_features.csv")

# (Optional) inspect columns
print("Feature columns:", list(df.columns))