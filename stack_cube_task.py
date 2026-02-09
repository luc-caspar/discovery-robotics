#!/usr/bin/env
from dm_control import composer
from dm_control.composer import variation

from dm_control.entities.props import Primitive
from dm_control.entities.manipulators.kinova.jaco_arm import JacoArm, JacoArmObservables
from dm_control.entities.manipulators.kinova.jaco_hand import JacoHand, JacoHandObservables
from dm_control.manipulation.shared.arenas import Standard
from dm_control.composer.observation import observable

import numpy as np
import mujoco.viewer as mjcv


class Jaco(composer.Entity):
    """
    Defines a full Jaco manipulator made of an arm and accompanying hand.
    """

    def _build(self, name=None, pinch_as_tcp=False):
        # Instantiate both arm and hand
        self._arm = JacoArm(name=name)
        self._hand = JacoHand(name=None if name is None else f'{name}_hand', use_pinch_site_as_tcp=pinch_as_tcp)

        # Attach the hand to the wrist
        self._arm.attach(self._hand)

    def _build_observables(self):
        return JacoObservables(self)

    @property
    def arm(self):
        return self._arm

    @property
    def hand(self):
        return self._hand

    @property
    def mjcf_model(self):
        return self._arm.mjcf_model


class JacoObservables(composer.Observables):
    """
    Combines the observables from both the Jaco hand and Jaco wrist.
    """

    @composer.observable
    def arm_joint_pos(self):
        return self._entity.arm.observables.joints_pos

    @composer.observable
    def arm_joint_vel(self):
        return self._entity.arm.observables.joints_vel

    @composer.observable
    def arm_joint_torque(self):
        return self._entity.arm.observables.joints_torque

    @composer.observable
    def hand_joint_pos(self):
        return self._entity.hand.observables.joints_pos

    @composer.observable
    def hand_joint_vel(self):
        return self._entity.hand.observables.joints_vel

    @composer.observable
    def hand_pinch_site_pos(self):
        return self._entity.hand.observables.pinch_site_pos

    @composer.observable
    def hand_pinch_site_rmat(self):
        return self._entity.hand.observables.pinch_site_rmat


class Cube(Primitive):
    """
    Defines a cube primitive, with observable position, orientation, and velocities.
    """

    def _build(self, size, name=None, mass=None, rgba=(0.5, 0.5, 0.5, 1)):
        # /!\ size is the half-size along each axis
        # Translate integer size into tuple
        if not isinstance(size, tuple):
            size = (size, ) * 3

        # Delegate the instantiation of the actual entity and its observables to the parent class
        if mass is not None:
            super()._build(geom_type='box', size=size, name=name, rgba=rgba, mass=mass)
        else:
            super()._build(geom_type='box', size=size, name=name, rgba=rgba)


class StackCubeTask(composer.Task):
    """
    Define an environment with two robot arms and two cubes.
    The task's goal is to stack the movable cube on the target one.
    """

    def __init__(self, cube_size, cube_mass):
        # Instantiate arena
        self._arena = Standard()

        # Instantiate both manipulator arms
        self._manip_left = Jaco(name='left', pinch_as_tcp=True)
        self._manip_right = Jaco(name='right', pinch_as_tcp=True)

        # Instantiate target and movable cubes
        self._cube_tgt = Cube(0.01, name='target', rgba=(1, 0, 0, 1))
        self._cube_mv = Cube(0.01, name='movable', rgba=(0, 1, 0, 1))

        # Attach all entities to the arena
        self._arena.attach(self._manip_left)
        self._arena.attach(self._manip_right)
        self._arena.attach(self._cube_tgt)  # Target cube is immovable
        self._arena.add_free_entity(self._cube_mv)  # Movable cube is movable

        # TODO: Configure arms' position, and random initial orientation
        # TODO: Configure initial poses based on defined Variation for cubes

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # TODO: Configure and enable observables
        self._observables = {}
        self._manip_left.observables.enable_all()
        self._observables['manip_left'] = self._manip_left.observables
        self._manip_right.observables.enable_all()
        self._observables['manip_right'] = self._manip_right.observables
        self._cube_tgt.observables.enable_all()
        self._observables['cube_tgt'] = self._cube_tgt.observables
        self._cube_mv.observables.enable_all()
        self._observables['cube_mv'] = self._cube_mv.observables

        for obs in self._observables.values():
            obs.enabled = True

        # TODO: Configure and enable task observables based on callable
        def eg_observables(physics):
            return 0

        self._task_observables = {'eg': observable.Generic(eg_observables)}
        self._task_observables['eg'].enabled = True


    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        # TODO: Randomize positions of cubes within arms' work area
        # TODO: Set initial position and orientation of manipulators
        self._manip_left.set_pose(physics, (0.5, 0.5, 0))
        self._manip_right.set_pose(physics, (-0.5, -0.5, 0))

        # TODO: Set initial position of cubes
        self._cube_tgt.set_pose(physics, (0, 0, 0))
        self._cube_mv.set_pose(physics, (0.75, 0.25, 0))

    def get_reward(self, physics):
        # TODO: define reward function
        return 0


if __name__ == "__main__":
    task = StackCubeTask(1, 30)
    env = composer.Environment(task, random_state=np.random.RandomState(42))
    env.reset()

    # Extract the underlying MuJoCo data and model structures necessary for visualization
    mjc_data = env.physics.data._data
    mjc_model = env.physics.model._model

    # Launch the mujoco viewer (this will run the simulation from keyframe 0)
    mjcv.launch(mjc_model, mjc_data)
