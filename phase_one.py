import os
import requests
import googleapiclient.discovery
import sys
from pathvalidate import sanitize_filename
import pathlib
from PIL import Image

def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} PLAYLIST_URL", file=sys.stderr)
        sys.exit(-1)
    
    playlist_url = sys.argv[1].split("list=")[-1]
    API_KEY = "AIzaSyC-KagZHFHpPo4vCnqWMfHBlOmn7F5DrKM"
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey = API_KEY)
    output = pathlib.Path("output/")
    output.mkdir(parents=True, exist_ok=True)

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
        thumb_url = video["snippet"]["thumbnails"]["default"]["url"]
        if "medium" in video["snippet"]["thumbnails"]:
            thumb_url = video["snippet"]["thumbnails"]["medium"]["url"]
        thumb_response = requests.get(thumb_url)
        artist_title = video["snippet"]["title"]
        if thumb_response.status_code == 200:
            filename = f"output/{sanitize_filename(artist_title)}.jpg"
            with open(filename, 'wb') as f:
                f.write(thumb_response.content)
            # crop image
            with Image.open(filename) as img:
                width, height = img.size
                new_size = min(width, height)
                # calculate coordinates
                left = (width - new_size) / 2
                top = (height - new_size) / 2
                right = (width + new_size) / 2
                bottom = (height + new_size) / 2
                img_cropped = img.crop((left, top, right, bottom))
                img_cropped.save(filename)
        else:
            print(f"Failed to download thumbnail for {artist_title}")
            return None

if __name__ == '__main__':
    main()