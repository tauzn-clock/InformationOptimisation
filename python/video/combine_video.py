ffmpeg -framerate 12 -i /scratchdata/test/combined/%0d.png -c:v libx264 -pix_fmt yuv420p output.mp4
