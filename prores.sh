cd output
ffmpeg -i cfd_%10d.png -c:v prores_ks -profile:3 -qscale:0 cfd.mov
cd ../
