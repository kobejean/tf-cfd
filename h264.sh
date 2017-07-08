cd output
# ffmpeg -framerate 60 -i cfd_%10d.png -c:v libx264 -preset veryslow -crf 0 cfd.mkv # 60fps
ffmpeg -i cfd_%10d.png -c:v libx264 -preset veryslow -crf 0 cfd.mkv
cd ../
