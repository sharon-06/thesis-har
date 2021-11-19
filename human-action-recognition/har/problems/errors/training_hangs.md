# Fix
The problem here had been corrupt video files.

I was using `from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip` but one should rather use `from moviepy.editor import VideoFileClip` as described [here](https://github.com/open-mmlab/mmaction2/issues/564#issuecomment-830618091).

In this way, one can filter those parts of the corrupt videos from generating clips.

Looks like the corrupt videos are the Japanese ones at the root, `j#01_1 - j#25_1`.

# Mentioned in issue

https://github.com/open-mmlab/mmaction2/issues/838

https://github.com/open-mmlab/mmaction2/issues/421

# Error Log
Training hangs after it has finished 5 epochs
