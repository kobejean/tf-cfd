cd output
# ffmpeg -framerate 60 -i cfd_%10d.png -c:v prores_ks -profile:3 -qscale:0 cfd.mov # 60fps
ffmpeg -i cfd_%10d.png -c:v prores_ks -profile:3 -qscale:0 cfd.mov # default 25fps
cd ../
