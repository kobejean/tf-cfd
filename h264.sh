cd output
ffmpeg -i cfd_%10d.png -c:v libx264 -preset veryslow -crf 0 cfd.mkv
cd ../
