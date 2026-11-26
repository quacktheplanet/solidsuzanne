#!/bin/sh
# Joins the composited segments (WORK/segments/*.mp4, in SEGMENTS order) and adds WORK/score.wav.
# Usage: sh assemble.sh OUTPUT.mp4 s1 s2 s3 s4
set -e
OUTFILE=$(realpath -m "$1"); shift
cd "${WORK:-work}"
: > concat.txt
for s in "$@"; do printf "file 'segments/%s.mp4'\n" "$s" >> concat.txt; done
ffmpeg -y -loglevel error -f concat -safe 0 -i concat.txt -c copy video.tmp.mp4 && mv video.tmp.mp4 video.mp4
ffmpeg -y -loglevel error -i video.mp4 -i score.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k \
  -shortest -movflags +faststart "${OUTFILE%.mp4}.tmp.mp4" && mv "${OUTFILE%.mp4}.tmp.mp4" "$OUTFILE"
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,r_frame_rate -of compact "$OUTFILE"
