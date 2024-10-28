from pytubefix import YouTube

YouTube('https://www.youtube.com/watch?v=zHMd5kTNljE').streams.get_highest_resolution().download()
