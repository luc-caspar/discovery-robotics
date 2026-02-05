#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from dm_control import mujoco, mjcf
import mujoco.viewer as mjcv
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib


# Set the rendering backend 'egl' or 'glfw'
os.environ['MUJOCO_GL'] = 'egl'
# Set GPU device to use for rendering
os.environ['MUJOCO_EGL_DEVICE_ID'] = "0"


# Constants
PWD = Path(__file__).parent
DATA_DIR = PWD.joinpath('Data')


if __name__ == "__main__":
    if False:
        # Define static MJCF model
        swinging_body = """
        <mujoco>
          <worldbody>
            <light name="top" pos="0 0 1"/>
            <body name="box_and_sphere" euler="0 0 -30">
                <!-- Joint is necessary for bodies to actually move in the world -->
                <!-- If movement is desired without being attached to another body, then use the 'free' joint type -->
                <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2" />
                <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
                <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """

        # Physics seems to be the top object gathering both MjModel and MjData below it
        # Both Model and Data are available as properties (i.e.: physics.model, physics.data)
        physics = mujoco.Physics.from_xml_string(swinging_body)

        model = physics.model
        data = physics.data

        named_model = physics.named.model
        named_data = physics.named.data

        physics.reset()  # Reset state and time
        print(model.geom_pos)  # returns a list of the positions of all geometries
        print(model.opt.gravity)
        print(data.time, data.qpos, data.qvel)  # qpos and qvel are supposed to return quaternions, but only a single value is printed?
        print(data.geom_xpos)

        # Named indexing makes it easy to access the exact data you want through memorable indexes
        print(named_model.geom_pos)
        print(named_data.geom_xpos)
        print(named_data.qpos['swing'])

        # It is possible to mix named indexing with numpy-like slices
        print(named_model.geom_rgba['red_box', :3])
        named_model.geom_rgba['red_box', :3] = np.random.rand(3)
        print(named_model.geom_rgba['red_box', :3])

        # Reset context needed to synchronize data object
        named_data.qpos['swing'] = np.pi
        print(f'Without reset: {named_data.geom_xpos["green_sphere", ["z"]]}')
        with physics.reset_context():
            named_data.qpos['swing'] = np.pi
        print(f'With reset: {named_data.geom_xpos["green_sphere", ["z"]]}')

        # Options for scene rendering
        scene_opts = mujoco.wrapper.core.MjvOption()
        scene_opts.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

        dur = 2  # seconds
        fps = 30

        frames = []
        physics.reset()  # Reset state and time

        while physics.data.time < dur:
            physics.step()  # Move forward in time but by how much? Are steps unitless as well?
            if len(frames) < physics.data.time * fps:
                pixels = physics.render(scene_option=scene_opts)
                frames.append(pixels)

        for idx, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(DATA_DIR.joinpath(f'frame_{idx}.jpg'))

    tippe_top = """
    <mujoco model="tippe top">
        <option integrator="RK4"/>
        <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
        </asset>
        <worldbody>
            <geom size=".2 .2 .01" type="plane" material="grid"/>
            <light pos="0 0 .6"/>
            <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
            <body name="top" pos="0 0 .02">
                <freejoint/>
                <geom name="ball" type="sphere" size=".02" />
                <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
                <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015" contype="0" conaffinity="0" group="3"/>
            </body>
        </worldbody>
        <keyframe>
            <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
        </keyframe>
    </mujoco>
    """

    physics = mujoco.Physics.from_xml_string(tippe_top)
    dur = 5
    fps = 60
    
    timevals = []
    ang_vel = []
    lin_vel = []
    stem_height = []
    frames = []

    physics.reset(0)  # Reset to the keyframe defined in the tippe_top string (i.e.: load a known state)
    data = physics.data

    # Extract the underlying MuJoCo data and model structures necessary for visualization
    mjc_data = physics.data._data
    mjc_model = physics.model._model

    # Launch the mujoco viewer (this will run the simulation from keyframe 0)
    mjcv.launch(model._model, data._data)

    exit()

    while physics.data.time < dur:
        physics.step()
        timevals.append(data.time)
        ang_vel.append(data.qvel[3:6].copy())  # Copy() is important since qvel points to a memory location which will change
        lin_vel.append(data.qvel[0:3].copy())  # Copy() is important since qvel points to a memory location which will change
        stem_height.append(physics.named.data.geom_xpos['stem', 'z'])
        if len(frames) < physics.data.time * fps:
            frames.append(Image.fromarray(physics.render(camera_id='closeup')))

    for idx, frame in enumerate(frames):
        frame.save(DATA_DIR.joinpath(f'frame_{idx}.jpg'))

    dpi = 100
    width = 480
    height = 640
    figsize = (width / dpi, height / dpi)
    _, ax = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=True)
    ax[0].plot(timevals, ang_vel)
    ax[0].set_title('angular velocity')
    ax[0].set_ylabel('radians / second')

    ax[1].plot(timevals, lin_vel)
    ax[1].set_title('linear velocity')
    ax[1].set_ylabel('meter / second')

    ax[2].plot(timevals, stem_height)
    ax[2].set_title('stem height')
    ax[2].set_xlabel('time (seconds)')
    ax[2].set_ylabel('meter')
    plt.show()
