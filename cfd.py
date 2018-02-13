#
#  cfd.py
#  tf-cfd
#
#  Created by Jean Flaherty on 7/1/17.
#  Copyright © 2017 kobejean. All rights reserved.
#
"""
script that runs cfd and produces a sequence of images animating the simulation
"""

import tensorflow as tf
import numpy as np
import math, os
from threading import Thread

# create output directory
if not os.path.exists("output"):
    os.makedirs("output")

LOGGING = False
PERIOD = 100

# # Dimensions (height x width)
DIM = (240, 600)
# DIM = (480, 1200)
# DIM = (1600, 2560) # Okar dimentions
# DIM = (1024, 2560) # Okar widescreen
# DIM = (2160, 3840) # 4k
# DIM = (1536, 3840) # widescreen 4k
# DIM = (1920, 4800)

velocity = 0.050
viscocity = 0.020


# useful constants
v = velocity # short hand
four9ths = 4.0 / 9.0
one9th = 1.0 / 9.0
one36th = 1.0 / 36.0
zeroes = tf.zeros(shape=DIM)
four9ths = 4.0 / 9.0
one9th = 1.0 / 9.0
one36th = 1.0 / 36.0

# circle barrier in the middle
barrier  = np.empty(shape=DIM, dtype=bool)
# fountain = np.empty(shape=DIM, dtype=float)
for i in range(DIM[0]):
    for j in range(DIM[1]):
        reli = DIM[0]/2.0 - i;
        relj = DIM[1]/2.0 - j;
        r = math.sqrt(reli*reli + relj*relj)
        barrier[i][j]  = (r < min(DIM[0], DIM[1]) * 0.2)
        # fountain[i][j] = 0.005 if (r < min(DIM[0], DIM[1]) * 0.3 and not barrier[i][j]) else 0.0


with tf.name_scope('variables') as scope:
    # variable initial values
    n0  = tf.fill(DIM, four9ths * (1.0 - 1.5*v*v)    )
    nE  = tf.fill(DIM, one9th * (1.0 + 3*v + 3*v*v)  )
    nW  = tf.fill(DIM, one9th * (1.0 - 3*v + 3*v*v)  )
    nN  = tf.fill(DIM, one9th * (1.0 - 1.5*v*v)      )
    nS  = tf.fill(DIM, one9th * (1.0 - 1.5*v*v)      )
    nNE = tf.fill(DIM, one36th * (1.0 + 3*v + 3*v*v) )
    nSE = tf.fill(DIM, one36th * (1.0 + 3*v + 3*v*v) )
    nNW = tf.fill(DIM, one36th * (1.0 - 3*v + 3*v*v) )
    nSW = tf.fill(DIM, one36th * (1.0 - 3*v + 3*v*v) )
    # variables (masked with barrier)
    n0  = tf.Variable(tf.where(barrier, zeroes, n0)  , name="n0" )
    nE  = tf.Variable(tf.where(barrier, zeroes, nE)  , name="nE" )
    nW  = tf.Variable(tf.where(barrier, zeroes, nW)  , name="nW" )
    nN  = tf.Variable(tf.where(barrier, zeroes, nN)  , name="nN" )
    nS  = tf.Variable(tf.where(barrier, zeroes, nS)  , name="nS" )
    nNE = tf.Variable(tf.where(barrier, zeroes, nNE) , name="nNE")
    nSE = tf.Variable(tf.where(barrier, zeroes, nSE) , name="nSE")
    nNW = tf.Variable(tf.where(barrier, zeroes, nNW) , name="nNW")
    nSW = tf.Variable(tf.where(barrier, zeroes, nSW) , name="nSW")

with tf.name_scope('computed_variables') as scope:
    # computed variables
    density = tf.fill(DIM, 1.0 )
    xvel    = tf.fill(DIM, v   )
    yvel    = tf.fill(DIM, 0.0 )
    speed2  = tf.fill(DIM, v*v )
    # computer variables (masked with barrier)
    density = tf.Variable(tf.where(barrier, zeroes, density) , name="density" )
    xvel    = tf.Variable(tf.where(barrier, zeroes, xvel)    , name="xvel"    )
    yvel    = tf.Variable(tf.where(barrier, zeroes, yvel)    , name="yvel"    )
    speed2  = tf.Variable(tf.where(barrier, zeroes, speed2)  , name="speed2"  )

with tf.name_scope('image') as scope:
    H = tf.fill(DIM, 0.5)
    S = tf.fill(DIM, 1.0)
    # V = tf.minimum(tf.sqrt(yvel)*6.0, 1.0)
    # V = tf.minimum(tf.sqrt(xvel)*6.0, 1.0)
    V = tf.minimum(tf.sqrt(speed2)*8.0, 1.0)
    HSV = tf.stack([H,S,V], axis=-1)
    RGB = tf.image.hsv_to_rgb(HSV, "RGB")*(2**16-1)
    RGB = tf.cast(RGB, dtype=tf.uint16)
    encoded_image = tf.image.encode_png(RGB)


# distribute the moving densities
def collide():
    with tf.name_scope('collide') as scope:
        omega = 1 / (3*viscocity + 0.5)
        n = n0 + nN + nS + nE + nW + nNW + nNE + nSW + nSE
        n = tf.where(barrier, zeroes, n)
        one9thn = one9th * n
        one36thn = one36th * n
        vx = (nE + nNE + nSE - nW - nNW - nSW) / n
        vx = tf.where(tf.greater(n, zeroes), vx, zeroes)
        vy = (nS + nSE + nSW - nN - nNE - nNW) / n
        vy = tf.where(tf.greater(n, zeroes), vy, zeroes)

        vx3 = 3 * vx
        vy3 = 3 * vy
        vx2 = vx * vx
        vy2 = vy * vy
        vxvy2 = 2 * vx * vy
        v2 = vx2 + vy2
        v215 = 1.5 * v2

        tmp_n0  = omega * (four9ths*n * (1                              - v215) - n0 )
        tmp_nE  = omega * (   one9thn * (1 + vx3       + 4.5*vx2        - v215) - nE )
        tmp_nW  = omega * (   one9thn * (1 - vx3       + 4.5*vx2        - v215) - nW )
        tmp_nN  = omega * (   one9thn * (1 - vy3       + 4.5*vy2        - v215) - nN )
        tmp_nS  = omega * (   one9thn * (1 + vy3       + 4.5*vy2        - v215) - nS )
        tmp_nNE = omega * (  one36thn * (1 + vx3 - vy3 + 4.5*(v2-vxvy2) - v215) - nNE)
        tmp_nNW = omega * (  one36thn * (1 - vx3 - vy3 + 4.5*(v2+vxvy2) - v215) - nNW)
        tmp_nSE = omega * (  one36thn * (1 + vx3 + vy3 + 4.5*(v2+vxvy2) - v215) - nSE)
        tmp_nSW = omega * (  one36thn * (1 - vx3 + vy3 + 4.5*(v2-vxvy2) - v215) - nSW)

        ops = tf.group( tf.assign(density , n ),
                        tf.assign(xvel    , vx),
                        tf.assign(yvel    , vy),
                        tf.assign(speed2  , v2),
                        tf.assign_add(n0 , tmp_n0 ),
                        tf.assign_add(nE , tmp_nE ),
                        tf.assign_add(nW , tmp_nW ),
                        tf.assign_add(nN , tmp_nN ),
                        tf.assign_add(nS , tmp_nS ),
                        tf.assign_add(nNE, tmp_nNE),
                        tf.assign_add(nSE, tmp_nSE),
                        tf.assign_add(nNW, tmp_nNW),
                        tf.assign_add(nSW, tmp_nSW))
    return ops

# stream densities
def stream():
    with tf.name_scope('stream') as scope:
        # set all density values at barrier sites to 0 before stream
        # density values that flow into the barrier will be used for bounce
        tmp_n0  = tf.where(barrier, zeroes, n0 )
        tmp_nE  = tf.where(barrier, zeroes, nE )
        tmp_nW  = tf.where(barrier, zeroes, nW )
        tmp_nN  = tf.where(barrier, zeroes, nN )
        tmp_nS  = tf.where(barrier, zeroes, nS )
        tmp_nNE = tf.where(barrier, zeroes, nNE)
        tmp_nSE = tf.where(barrier, zeroes, nSE)
        tmp_nNW = tf.where(barrier, zeroes, nNW)
        tmp_nSW = tf.where(barrier, zeroes, nSW)

        # stream by slicing and padding matrices in appropriate directions
        # slice sizes
        Sx  = (DIM[0]  , DIM[1]-1)
        Sy  = (DIM[0]-2, DIM[1]  )
        Sxy = (DIM[0]-2, DIM[1]-1)
        tmp_n0  = tf.slice(tmp_n0 , [1,0], Sy)
        tmp_nE  = tf.slice(tmp_nE , [1,0], Sxy)
        tmp_nW  = tf.slice(tmp_nW , [1,1], Sxy)
        tmp_nN  = tf.slice(tmp_nN , [2,0], Sy )
        tmp_nS  = tf.slice(tmp_nS , [0,0], Sy )
        tmp_nNE = tf.slice(tmp_nNE, [2,0], Sxy)
        tmp_nSE = tf.slice(tmp_nSE, [0,0], Sxy)
        tmp_nNW = tf.slice(tmp_nNW, [2,1], Sxy)
        tmp_nSW = tf.slice(tmp_nSW, [0,1], Sxy)

        # padding
        PX = (DIM[0]  , 1       )
        Px = (DIM[0]-2, 1       )
        PY = (1       , DIM[1]  )
        Py = (1       , DIM[1]-1)

        # pad_n0x  = tf.fill(Px, four9ths * (1 - 1.5*v*v)     )
        pad_n0y  = tf.fill(PY, four9ths * (1 - 1.5*v*v)     )

        pad_nEx  = tf.fill(Px, one9th * (1 + 3*v + 3*v*v)   )
        pad_nEy  = tf.fill(PY, one9th * (1 + 3*v + 3*v*v)   )

        pad_nWx  = tf.fill(Px, one9th * (1 - 3*v + 3*v*v)   )
        pad_nWy  = tf.fill(PY, one9th * (1 - 3*v + 3*v*v)   )

        # pad_nNx  = tf.fill(Px, one9th * (1 - 1.5*v*v)       )
        pad_nNy  = tf.fill(PY, one9th * (1 - 1.5*v*v)       )

        # pad_nSx  = tf.fill(Px, one9th * (1 - 1.5*v*v)       )
        pad_nSy  = tf.fill(PY, one9th * (1 - 1.5*v*v)       )

        pad_nNEx = tf.fill(Px, one36th * (1 + 3*v + 3*v*v)  )
        pad_nNEy = tf.fill(PY, one36th * (1 + 3*v + 3*v*v)  )

        pad_nSEx = tf.fill(Px, one36th * (1 + 3*v + 3*v*v)  )
        pad_nSEy = tf.fill(PY, one36th * (1 + 3*v + 3*v*v)  )

        pad_nNWx = tf.fill(Px, one36th * (1 - 3*v + 3*v*v)  )
        pad_nNWy = tf.fill(PY, one36th * (1 - 3*v + 3*v*v)  )

        pad_nSWx = tf.fill(Px, one36th * (1 - 3*v + 3*v*v)  )
        pad_nSWy = tf.fill(PY, one36th * (1 - 3*v + 3*v*v)  )

        # tmp_n0  = tf.concat([pad_n0x , tmp_n0  , pad_n0x ], 1)
        tmp_n0  = tf.concat([pad_n0y , tmp_n0  , pad_n0y ], 0)

        tmp_nE  = tf.concat([pad_nEx , tmp_nE  ], 1)
        tmp_nE  = tf.concat([pad_nEy , tmp_nE  , pad_nEy ], 0)

        tmp_nW  = tf.concat([tmp_nW  , pad_nWx ], 1)
        tmp_nW  = tf.concat([pad_nWy , tmp_nW  , pad_nWy ], 0)

        # tmp_nN  = tf.concat([pad_nNx , tmp_nN  , pad_nNx ], 1)
        tmp_nN  = tf.concat([pad_nNy , tmp_nN  , pad_nNy ], 0)

        # tmp_nS  = tf.concat([pad_nSx , tmp_nS  , pad_nSx ], 1)
        tmp_nS  = tf.concat([pad_nSy , tmp_nS  , pad_nSy ], 0)

        tmp_nNE = tf.concat([pad_nNEx, tmp_nNE ], 1)
        tmp_nNE = tf.concat([pad_nNEy, tmp_nNE , pad_nNEy], 0)

        tmp_nSE = tf.concat([pad_nSEx, tmp_nSE ], 1)
        tmp_nSE = tf.concat([pad_nSEy, tmp_nSE , pad_nSEy], 0)

        tmp_nNW = tf.concat([tmp_nNW , pad_nNWx], 1)
        tmp_nNW = tf.concat([pad_nNWy, tmp_nNW , pad_nNWy], 0)

        tmp_nSW = tf.concat([tmp_nSW , pad_nSWx], 1)
        tmp_nSW = tf.concat([pad_nSWy, tmp_nSW , pad_nSWy], 0)

        ops = tf.group( tf.assign(n0 , tmp_n0 ),
                        tf.assign(nE , tmp_nE ),
                        tf.assign(nW , tmp_nW ),
                        tf.assign(nN , tmp_nN ),
                        tf.assign(nS , tmp_nS ),
                        tf.assign(nNE, tmp_nNE),
                        tf.assign(nSE, tmp_nSE),
                        tf.assign(nNW, tmp_nNW),
                        tf.assign(nSW, tmp_nSW))
    return ops

# bounce off the barrier (flip direction)
def bounce():
    with tf.name_scope('bounce') as scope:

        bool_nW  = tf.logical_and(barrier, tf.greater(nW ,zeroes))
        bool_nE  = tf.logical_and(barrier, tf.greater(nE ,zeroes))
        bool_nS  = tf.logical_and(barrier, tf.greater(nS ,zeroes))
        bool_nN  = tf.logical_and(barrier, tf.greater(nN ,zeroes))
        bool_nSW = tf.logical_and(barrier, tf.greater(nSW,zeroes))
        bool_nNW = tf.logical_and(barrier, tf.greater(nNW,zeroes))
        bool_nSE = tf.logical_and(barrier, tf.greater(nSE,zeroes))
        bool_nNE = tf.logical_and(barrier, tf.greater(nNE,zeroes))

        dif_nE  = tf.where(bool_nW , nW , zeroes)
        dif_nW  = tf.where(bool_nE , nE , zeroes)
        dif_nN  = tf.where(bool_nS , nS , zeroes)
        dif_nS  = tf.where(bool_nN , nN , zeroes)
        dif_nNE = tf.where(bool_nSW, nSW, zeroes)
        dif_nSE = tf.where(bool_nNW, nNW, zeroes)
        dif_nNW = tf.where(bool_nSE, nSE, zeroes)
        dif_nSW = tf.where(bool_nNE, nNE, zeroes)

        # slice sizes
        Sx  = (DIM[0]  , DIM[1]-1)
        Sy  = (DIM[0]-2, DIM[1]  )
        Sxy = (DIM[0]-2, DIM[1]-1)
        # dif_n0  = tf.slice(n0 , [1,0], Sy)
        dif_nE  = tf.slice(dif_nE , [1,0], Sxy)
        dif_nW  = tf.slice(dif_nW , [1,1], Sxy)
        dif_nN  = tf.slice(dif_nN , [2,0], Sy )
        dif_nS  = tf.slice(dif_nS , [0,0], Sy )
        dif_nNE = tf.slice(dif_nNE, [2,0], Sxy)
        dif_nSE = tf.slice(dif_nSE, [0,0], Sxy)
        dif_nNW = tf.slice(dif_nNW, [2,1], Sxy)
        dif_nSW = tf.slice(dif_nSW, [0,1], Sxy)

        # padding
        PX = (DIM[0]  , 1       )
        Px = (DIM[0]-2, 1       )
        PY = (1       , DIM[1]  )
        Py = (1       , DIM[1]-1)

        # pad_n0x  = tf.fill(Px, 0.0)
        # pad_n0y  = tf.fill(PY, 0.0)

        pad_nEx  = tf.fill(Px, 0.0)
        pad_nEy  = tf.fill(PY, 0.0)

        pad_nWx  = tf.fill(Px, 0.0)
        pad_nWy  = tf.fill(PY, 0.0)

        # pad_nNx  = tf.fill(Px, 0.0)
        pad_nNy  = tf.fill(PY, 0.0)

        # pad_nSx  = tf.fill(Px, 0.0)
        pad_nSy  = tf.fill(PY, 0.0)

        pad_nNEx = tf.fill(Px, 0.0)
        pad_nNEy = tf.fill(PY, 0.0)

        pad_nSEx = tf.fill(Px, 0.0)
        pad_nSEy = tf.fill(PY, 0.0)

        pad_nNWx = tf.fill(Px, 0.0)
        pad_nNWy = tf.fill(PY, 0.0)

        pad_nSWx = tf.fill(Px, 0.0)
        pad_nSWy = tf.fill(PY, 0.0)

        # dif_n0  = tf.concat([pad_n0x , dif_n0  , pad_n0x ], 1)
        # dif_n0  = tf.concat([pad_n0y , dif_n0  , pad_n0y ], 0)

        dif_nE  = tf.concat([pad_nEx , dif_nE  ], 1)
        dif_nE  = tf.concat([pad_nEy , dif_nE  , pad_nEy ], 0)

        dif_nW  = tf.concat([dif_nW  , pad_nWx ], 1)
        dif_nW  = tf.concat([pad_nWy , dif_nW  , pad_nWy ], 0)

        # dif_nN  = tf.concat([pad_nNx , dif_nN  , pad_nNx ], 1)
        dif_nN  = tf.concat([pad_nNy , dif_nN  , pad_nNy ], 0)

        # dif_nS  = tf.concat([pad_nSx , dif_nS  , pad_nSx ], 1)
        dif_nS  = tf.concat([pad_nSy , dif_nS  , pad_nSy ], 0)

        dif_nNE = tf.concat([pad_nNEx, dif_nNE ], 1)
        dif_nNE = tf.concat([pad_nNEy, dif_nNE , pad_nNEy], 0)

        dif_nSE = tf.concat([pad_nSEx, dif_nSE ], 1)
        dif_nSE = tf.concat([pad_nSEy, dif_nSE , pad_nSEy], 0)

        dif_nNW = tf.concat([dif_nNW , pad_nNWx], 1)
        dif_nNW = tf.concat([pad_nNWy, dif_nNW , pad_nNWy], 0)

        dif_nSW = tf.concat([dif_nSW , pad_nSWx], 1)
        dif_nSW = tf.concat([pad_nSWy, dif_nSW , pad_nSWy], 0)

        ops = tf.group( tf.assign_add(nE , dif_nE ),
                        tf.assign_add(nW , dif_nW ),
                        tf.assign_add(nN , dif_nN ),
                        tf.assign_add(nS , dif_nS ),
                        tf.assign_add(nNE, dif_nNE),
                        tf.assign_add(nSE, dif_nSE),
                        tf.assign_add(nNW, dif_nNW),
                        tf.assign_add(nSW, dif_nSW))
    return ops


# three steps per time step
collide_step = collide()
stream_step = stream()
bounce_step = bounce()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    if LOGGING:
        writer = tf.summary.FileWriter("log/", sess.graph)

    for t in range(10000000000):
        print("T: {}".format(t), end="\r")
        sess.run(collide_step)
        sess.run(stream_step)
        sess.run(bounce_step)
        if t % PERIOD == 0:
            # print(sess.run(RGB))
            image = sess.run(encoded_image)
            outpath = "output/cfd_{0:0>10}.png".format(t//PERIOD)

            def write(image, outpath):
                with open(outpath, 'wb') as f:
                    f.write(image)
            worker = Thread(target=write, args=(image, outpath,))
            worker.setDaemon(True)
            worker.start()
