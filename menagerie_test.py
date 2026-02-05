#!/usr/bin/env python3
if True:
    # This might be the easiest/cleanest and does not require the `robot_description` package
    from dm_control import mujoco
    import mujoco.viewer as mjcv

    # Load robot from file directly
    # In mujoco_menagerie repository, the mjx files seems to contain more realistic physics/properties (i.e.: deformation, dampening, ...)
    # The regular xml files are an idealized (i.e.: rigid) version of the robot
    physics = mujoco.Physics.from_xml_path('mujoco_menagerie/aloha/scene.xml')

    # Extract both MuJoCo data and model structures
    mjc_data = physics.data._data
    mjc_model = physics.model._model

    # Launch the interactive viewer
    mjcv.launch(mjc_model, mjc_data)

else:
    # This works as well, since providing the viewer with only the model will trigger the viewer to automatically compute the data
    import mujoco
    import mujoco.viewer as mjcv

    from robot_descriptions.loaders.mujoco import load_robot_description

    # Get the model from the menagerie
    mjc_model = load_robot_description('panda_mj_description')

    # Launch the viewer to explore the model
    mjcv.launch(mjc_model)

    # But I wanted to know if we could import a model from the menagerie directly into dm_control
    from dm_control import mujoco
    from dm_control.mujoco.wrapper import MjModel
    import mujoco.viewer as mjcv

    from robot_descriptions.loaders.mujoco import load_robot_description


    # Get the model from the menagerie
    mjc_model = load_robot_description('panda_mj_description')
    model = MjModel(mjc_model)

    # Load the model in dm_control to get the data (= state) computed automatically
    physics = mujoco.Physics.from_model(model)

    # Launch the viewer to explore the model
    mjc_data = physics.data._data
    mjcv.launch(mjc_model)

