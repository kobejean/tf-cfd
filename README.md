# tf-cfd
Computational fluid dynamics with tensorflow.
<p align="center">
    <img src="docs/media/screenshot.png"/>
</p>

### Dependancies
- tensorflow (nightly build or 1.6.0): https://www.tensorflow.org
- ffmpeg: https://www.ffmpeg.org/download.html

### Usage
Generate sequence of pngs by running cfd.py:
``` shell
$ python3 cfd.py
```
You can open the output folder with the [Sigma Unit](https://github.com/kobejean/sigma-unit) app to see live updates of the frames.

Encode video by running h264.sh or prores.sh:
``` shell
$ ./h264.sh
```
or
``` shell
$ ./prores.sh
```
