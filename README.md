# Tier Game
This is a simple program used to scrape YouTube thumbnails, as well as identify them within a tier list. It's intended for use in tier games, an original way to rank music and guess your friends' rankings!

# How the game works
Tier games consist of 3 rounds. The first is a submission round, where every participant sends the Gamemaster two songs. These songs are then shuffled and put together into a YouTube playlist, and the thumbnails are put into a Tiermaker tier list template, and both are released to the participants. This begins phase two, which simply involves each participants listening to and ranking the songs, typically with a deadline of 4-5 days after the list is released. The lists are submitted to the Gamemaster, who then randomizes them and posts the anonymized lists, beginning the third phase. Participants submit their guesses to the Gamemaster, who collects the answers and creates a chart of all the songs with their scores, as well as an answer key with each name tied to the initial list.

# What the program does
This program has two phases. The first is a simple thumbnail scraper that takes in a YouTube playlist URL and gets the thumbnails from each video. It squares them and downsizes them, which will become important for phase two. Phase two takes in a folder of all the Tiermaker list images, and it uses OpenCV to perform image recognition on them, aggregating the scores of each song into a spreadsheet at the end. This automates a lot of the work the Gamemaster normally has to do to create the phase 3 images, allowing the game to be run much easier.

# Usage
To use this program, you will need a YouTube API Key, which you can learn more about acquiring [here](https://developers.google.com/youtube/v3/getting-started). Run ```start.bat```, which will initialize the environment before prompting you for your API key. Once you've entered this, you will be brought to the main view of the program. Option one will require a YouTube URL, and outputs the squared images to the ```output``` folder, like below.
![image](https://github.com/user-attachments/assets/3fd47189-a501-41f4-ae6f-9a95e5de3ee6)
The Gamemaster should take these output images, upload them to [Tiermaker](https://tiermaker.com/categories/create/) as a new template, and release the playlist and template to the participants.


Option two on the main view will run image recognition on tier lists found in the ```input``` folder. The name of each image will be the corresponding name in the final spreadsheet, so set them accordingly based on who submitted them.
![image](https://github.com/user-attachments/assets/97ebfdea-6185-4f4c-92a7-d3e54e92f93a)


The final output will be sent to ```song_rankings.xlsx```.
