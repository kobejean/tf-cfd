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

print("INITIALIZING...")

# if using older version of tensorflow < 1.6.0
# manip = tf.load_op_library('user_ops/roll_op.so')

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

        tmp_nE  = tf.manip.roll(nE , shift=[ 0, 1], axis=[0,1])
        tmp_nW  = tf.manip.roll(nW , shift=[ 0,-1], axis=[0,1])
        tmp_nN  = tf.manip.roll(nN , shift=[-1, 0], axis=[0,1])
        tmp_nS  = tf.manip.roll(nS , shift=[ 1, 0], axis=[0,1])
        tmp_nNE = tf.manip.roll(nNE, shift=[-1, 1], axis=[0,1])
        tmp_nSE = tf.manip.roll(nSE, shift=[ 1, 1], axis=[0,1])
        tmp_nNW = tf.manip.roll(nNW, shift=[-1,-1], axis=[0,1])
        tmp_nSW = tf.manip.roll(nSW, shift=[ 1,-1], axis=[0,1])

        ops = tf.group(tf.assign(nE , tmp_nE ),
                       tf.assign(nW , tmp_nW ),
                       tf.assign(nN , tmp_nN ),
                       tf.assign(nS , tmp_nS ),
                       tf.assign(nNE, tmp_nNE),
                       tf.assign(nSE, tmp_nSE),
                       tf.assign(nNW, tmp_nNW),
                       tf.assign(nSW, tmp_nSW))
    return ops

# add force to stream
def force():
    with tf.name_scope('force') as scope:
        # padding
        PX = [DIM[0]]
        PY = [DIM[1]]

        # pad_n0x  = tf.fill(PX, four9ths * (1 - 1.5*v*v)     )
        pad_n0y  = tf.fill(PY, four9ths * (1 - 1.5*v*v)     )

        pad_nEx  = tf.fill(PX, one9th * (1 + 3*v + 3*v*v)   )
        pad_nEy  = tf.fill(PY, one9th * (1 + 3*v + 3*v*v)   )

        pad_nWx  = tf.fill(PX, one9th * (1 - 3*v + 3*v*v)   )
        pad_nWy  = tf.fill(PY, one9th * (1 - 3*v + 3*v*v)   )

        # pad_nNx  = tf.fill(PX, one9th * (1 - 1.5*v*v)       )
        pad_nNy  = tf.fill(PY, one9th * (1 - 1.5*v*v)       )

        # pad_nSx  = tf.fill(PX, one9th * (1 - 1.5*v*v)       )
        pad_nSy  = tf.fill(PY, one9th * (1 - 1.5*v*v)       )

        pad_nNEx = tf.fill(PX, one36th * (1 + 3*v + 3*v*v)  )
        pad_nNEy = tf.fill(PY, one36th * (1 + 3*v + 3*v*v)  )

        pad_nSEx = tf.fill(PX, one36th * (1 + 3*v + 3*v*v)  )
        pad_nSEy = tf.fill(PY, one36th * (1 + 3*v + 3*v*v)  )

        pad_nNWx = tf.fill(PX, one36th * (1 - 3*v + 3*v*v)  )
        pad_nNWy = tf.fill(PY, one36th * (1 - 3*v + 3*v*v)  )

        pad_nSWx = tf.fill(PX, one36th * (1 - 3*v + 3*v*v)  )
        pad_nSWy = tf.fill(PY, one36th * (1 - 3*v + 3*v*v)  )

        ops = tf.group( n0 [0       , ...].assign(pad_n0y ),
                        n0 [DIM[0]-1, ...].assign(pad_n0y ),
                        nE [...,        0].assign(pad_nEx ),
                        nE [0       , ...].assign(pad_nEy ),
                        nE [DIM[0]-1, ...].assign(pad_nEy ),
                        nW [..., DIM[1]-1].assign(pad_nWx ),
                        nW [0       , ...].assign(pad_nWy ),
                        nW [DIM[0]-1, ...].assign(pad_nWy ),
                        nN [0       , ...].assign(pad_nNy ),
                        nN [DIM[0]-1, ...].assign(pad_nNy ),
                        nS [0       , ...].assign(pad_nSy ),
                        nS [DIM[0]-1, ...].assign(pad_nSy ),
                        nNE[...,        0].assign(pad_nNEx),
                        nNE[0       , ...].assign(pad_nNEy),
                        nNE[DIM[0]-1, ...].assign(pad_nNEy),
                        nSE[...,        0].assign(pad_nSEx),
                        nSE[0       , ...].assign(pad_nSEy),
                        nSE[DIM[0]-1, ...].assign(pad_nSEy),
                        nNW[..., DIM[1]-1].assign(pad_nNWx),
                        nNW[0       , ...].assign(pad_nNWy),
                        nNW[DIM[0]-1, ...].assign(pad_nNWy),
                        nSW[..., DIM[1]-1].assign(pad_nSWx),
                        nSW[0       , ...].assign(pad_nSWy),
                        nSW[DIM[0]-1, ...].assign(pad_nSWy),)
    return ops

# bounce off the barrier (flip direction)
def bounce():
    with tf.name_scope('bounce') as scope:
        not_barrier = tf.logical_not(barrier)

        bool_nE  = tf.manip.roll(not_barrier, shift=[ 0, 1], axis=[0,1])
        bool_nW  = tf.manip.roll(not_barrier, shift=[ 0,-1], axis=[0,1])
        bool_nN  = tf.manip.roll(not_barrier, shift=[-1, 0], axis=[0,1])
        bool_nS  = tf.manip.roll(not_barrier, shift=[ 1, 0], axis=[0,1])
        bool_nNE = tf.manip.roll(not_barrier, shift=[-1, 1], axis=[0,1])
        bool_nSE = tf.manip.roll(not_barrier, shift=[ 1, 1], axis=[0,1])
        bool_nNW = tf.manip.roll(not_barrier, shift=[-1,-1], axis=[0,1])
        bool_nSW = tf.manip.roll(not_barrier, shift=[ 1,-1], axis=[0,1])

        bool_nE  = tf.logical_and(bool_nE , barrier)
        bool_nW  = tf.logical_and(bool_nW , barrier)
        bool_nN  = tf.logical_and(bool_nN , barrier)
        bool_nS  = tf.logical_and(bool_nS , barrier)
        bool_nNE = tf.logical_and(bool_nNE, barrier)
        bool_nSE = tf.logical_and(bool_nSE, barrier)
        bool_nNW = tf.logical_and(bool_nNW, barrier)
        bool_nSW = tf.logical_and(bool_nSW, barrier)

        bounce_nE  = tf.where(bool_nW , nW , zeroes)
        bounce_nW  = tf.where(bool_nE , nE , zeroes)
        bounce_nN  = tf.where(bool_nS , nS , zeroes)
        bounce_nS  = tf.where(bool_nN , nN , zeroes)
        bounce_nNE = tf.where(bool_nSW, nSW, zeroes)
        bounce_nSE = tf.where(bool_nNW, nNW, zeroes)
        bounce_nNW = tf.where(bool_nSE, nSE, zeroes)
        bounce_nSW = tf.where(bool_nNE, nNE, zeroes)

        bounce_nE  = tf.manip.roll(bounce_nE , shift=[ 0, 1], axis=[0,1])
        bounce_nW  = tf.manip.roll(bounce_nW , shift=[ 0,-1], axis=[0,1])
        bounce_nN  = tf.manip.roll(bounce_nN , shift=[-1, 0], axis=[0,1])
        bounce_nS  = tf.manip.roll(bounce_nS , shift=[ 1, 0], axis=[0,1])
        bounce_nNE = tf.manip.roll(bounce_nNE, shift=[-1, 1], axis=[0,1])
        bounce_nSE = tf.manip.roll(bounce_nSE, shift=[ 1, 1], axis=[0,1])
        bounce_nNW = tf.manip.roll(bounce_nNW, shift=[-1,-1], axis=[0,1])
        bounce_nSW = tf.manip.roll(bounce_nSW, shift=[ 1,-1], axis=[0,1])

        ops = tf.group( tf.assign_add(nE , bounce_nE ),
                        tf.assign_add(nW , bounce_nW ),
                        tf.assign_add(nN , bounce_nN ),
                        tf.assign_add(nS , bounce_nS ),
                        tf.assign_add(nNE, bounce_nNE),
                        tf.assign_add(nSE, bounce_nSE),
                        tf.assign_add(nNW, bounce_nNW),
                        tf.assign_add(nSW, bounce_nSW))
    return ops


# four steps per time step
def time():
    with tf.name_scope('time_step') as scope:
        collide_step = collide()
        force_step = force()
        with tf.control_dependencies([collide_step, force_step]):
            stream_step = stream()
        with tf.control_dependencies([stream_step]):
            bounce_step = bounce()
    return bounce_step

time_step = time()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    if LOGGING:
        writer = tf.summary.FileWriter("log/", sess.graph)

    for t in range(10000000000):
        print("T: {}".format(t), end="\r")
        sess.run(time_step)
        if t % PERIOD == 0:
            image = sess.run(encoded_image)
            outpath = "output/cfd_{0:0>10}.png".format(t//PERIOD)

            def write(image, outpath):
                with open(outpath, 'wb') as f:
                    f.write(image)
            worker = Thread(target=write, args=(image, outpath,))
            worker.setDaemon(True)
            worker.start()
