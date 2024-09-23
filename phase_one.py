import os
import requests
import googleapiclient.discovery
import sys
from pathvalidate import sanitize_filename

def download_thumbnail(youtube_url, artist_title):
    # Extract video ID from the URL
    video_id = youtube_url.split("v=")[-1]
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    
    # Download the thumbnail
    response = requests.get(thumbnail_url)
    if response.status_code == 200:
        filename = f"{artist_title}.jpg"
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    else:
        print(f"Failed to download thumbnail for {artist_title}")
        return None

# # Example usage
# thumbnails = []
# thumbnails.append(download_thumbnail("https://www.youtube.com/watch?v=VIDEO_ID", "Artist:Title"))
def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} PLAYLIST_URL", file=sys.stderr)
        sys.exit(-1)
    
    playlist_url = sys.argv[1].split("list=")[-1]
    API_KEY = "AIzaSyC-KagZHFHpPo4vCnqWMfHBlOmn7F5DrKM"
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey = API_KEY)

    request = youtube.playlistItems().list(
        part = "snippet",
        playlistId = playlist_url,
        maxResults = 50
    )
    response = request.execute()

    playlist_items = []
    while request is not None:
        response = request.execute()
        playlist_items += response["items"]
        request = youtube.playlistItems().list_next(request, response)

    # print(f"total: {len(playlist_items)}")
    # print(playlist_items)

    # get thumbnails
    for video in playlist_items:
        thumb_url = video["snippet"]["thumbnails"]["high"]["url"]
        if "maxres" in video["snippet"]["thumbnails"]:
            thumb_url = video["snippet"]["thumbnails"]["maxres"]["url"]
        thumb_response = requests.get(thumb_url)
        artist_title = video["snippet"]["title"]
        if thumb_response.status_code == 200:
            filename = f"{artist_title}.jpg"
            filename = sanitize_filename(filename)
            with open(filename, 'wb') as f:
                f.write(thumb_response.content)
        else:
            print(f"Failed to download thumbnail for {artist_title}")
            return None

if __name__ == '__main__':
    main()