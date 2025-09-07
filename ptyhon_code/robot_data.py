import numpy as np
import utility as ram 

class Robot:
    class Body:
         #constructor for Body
        def __init__(self, parent, name, pos, quat, ipos, iquat, mass, inertia, joint_axis, joint_range):
            self.parent = parent
            self.name = name
            self.pos = np.array(pos)
            self.quat = np.array(quat)
            self.ipos = np.array(ipos)
            self.iquat = np.array(iquat)
            self.mass = mass
            self.inertia = np.array(inertia)
            self.joint_axis = np.array(joint_axis)
            self.joint_range = np.array(joint_range)

           #Function to add body
    def add_body(self, body_id, parent, name, pos, quat, ipos, iquat, mass, inertia, joint_axis, joint_range):
        self.body[body_id] = Robot.Body(parent, name, pos, quat, ipos, iquat, mass, inertia, joint_axis, joint_range)

    class Params:
        def __init__(self):
            #From ur5e.xml
# <body name="base" quat="1 0 0 1" childclass="ur5e">
# <inertial mass="4" pos="0 0 0" diaginertia="0.00443333156 0.00443333156 0.0072"/>
# <geom mesh="base_0" material="black" class="visual"/>
# <geom mesh="base_1" material="jointgray" class="visual"/>

            base_quat = np.array([1, 0, 0, 1])
            self.base_quat = ram.quat_normalize(base_quat)

            self.end_eff_pos_local = np.array([0, 0.1, 0])
            end_eff_quat_local = np.array([-1, 1, 1, 1])
            self.end_eff_quat_local = ram.quat_normalize(end_eff_quat_local)

    #constructor for Robot
    def __init__(self):
        self.body = {}
        self.params = Robot.Params()  # Initialize robot parameters


robot = Robot()

#  <body name="shoulder_link" pos="0 0 0.163">
#         <inertial mass="3.7" pos="0 0 0" diaginertia="0.0102675 0.0102675 0.00666"/>
#         <joint name="shoulder_pan" axis="0 0 1"/>
#         <geom mesh="shoulder_0" material="urblue" class="visual"/>
#         <geom mesh="shoulder_1" material="black" class="visual"/>
#         <geom mesh="shoulder_2" material="jointgray" class="visual"/>
#         <geom class="collision" size="0.06 0.06" pos="0 0 -0.04" />

robot.add_body(
    1, parent='ground', name='shoulder_link', pos=[0, 0, 0.163],
    quat=[1, 0, 0, 0], ipos=[0, 0, 0], iquat=[1, 0, 0, 0],
    mass=3.7, inertia=[0.0102675, 0.0102675, 0.00666],
    joint_axis=[0, 0, 1], joint_range=[-6.28319, 6.28319]
)

# From ur5e.xml
# <body name="upper_arm_link" pos="0 0.138 0" quat="1 0 1 0">
# <inertial mass="8.393" pos="0 0 0.2125" diaginertia="0.133886 0.133886 0.0151074"/>
# <joint name="shoulder_lift"/>
# <geom mesh="upperarm_0" material="linkgray" class="visual"/>
# <geom mesh="upperarm_1" material="black" class="visual"/>
# <geom mesh="upperarm_2" material="jointgray" class="visual"/>
# <geom mesh="upperarm_3" material="urblue" class="visual"/>
# <geom class="collision" pos="0 -0.04 0" quat="1 1 0 0" size="0.06 0.06"/>
# <geom class="collision" size="0.05 0.2" pos="0 0 0.2"/>


robot.add_body(
    2, parent='shoulder_link', name='upper_arm_link', pos=[0, 0.138, 0],
    quat=[1, 0, 1, 0], ipos=[0, 0, 0.2125], iquat=[1, 0, 0, 0],
    mass=8.393, inertia=[0.133886, 0.133886, 0.0151074],
    joint_axis=[0, 1, 0], joint_range=[-6.28319, 6.28319]
)


# From ur5e.xml
# <body name="forearm_link" pos="0 -0.131 0.425">
# <inertial mass="2.275" pos="0 0 0.196" diaginertia="0.0311796 0.0311796 0.004095"/>
# <joint name="elbow" class="joint_limited"/>
# <geom mesh="forearm_0" material="urblue" class="visual"/>
# <geom mesh="forearm_1" material="linkgray" class="visual"/>
# <geom mesh="forearm_2" material="black" class="visual"/>
# <geom mesh="forearm_3" material="jointgray" class="visual"/>
# <geom class="collision" pos="0 0.08 0" quat="1 1 0 0" size="0.055 0.06"/>
# <geom class="collision" size="0.038 0.19" pos="0 0 0.2"/>

robot.add_body(
    3, parent='upper_arm_link', name='forearm_link', pos=[0, -0.131, 0.425],
    quat=[1, 0, 0, 0], ipos=[0, 0, 0.196], iquat=[1, 0, 0, 0],
    mass=2.275, inertia=[0.0311796, 0.0311796, 0.004095],
    joint_axis=[0, 1, 0], joint_range=[-6.28319, 6.28319]
)


# From ur5e.xml
# <body name="wrist_1_link" pos="0 0 0.392" quat="1 0 1 0">
# <inertial mass="1.219" pos="0 0.127 0" diaginertia="0.0025599 0.0025599 0.0021942"/>
# <joint name="wrist_1"/>
# <geom mesh="wrist1_0" material="black" class="visual"/>
# <geom mesh="wrist1_1" material="urblue" class="visual"/>
# <geom mesh="wrist1_2" material="jointgray" class="visual"/>
# <geom class="collision" pos="0 0.05 0" quat="1 1 0 0" size="0.04 0.07"/>


robot.add_body(
    4, parent='forearm_link', name='wrist_1_link', pos=[0, 0, 0.392],
    quat=[1, 0, 1, 0], ipos=[0, 0.127, 0], iquat=[1, 0, 0, 0],
    mass=1.219, inertia=[0.0025599, 0.0025599, 0.0021942],
    joint_axis=[0, 1, 0], joint_range=[-6.28319, 6.28319]
)

# From ur5e.xml
# <body name="wrist_2_link" pos="0 0.127 0">
# <inertial mass="1.219" pos="0 0 0.1" diaginertia="0.0025599 0.0025599 0.0021942"/>
# <joint name="wrist_2" axis="0 0 1"/>
# <geom mesh="wrist2_0" material="black" class="visual"/>
# <geom mesh="wrist2_1" material="urblue" class="visual"/>
# <geom mesh="wrist2_2" material="jointgray" class="visual"/>
# <geom class="collision" size="0.04 0.06" pos="0 0 0.04"/>
# <geom class="collision" pos="0 0.02 0.1" quat="1 1 0 0" size="0.04 0.04"/>

robot.add_body(
    5, parent='wrist_1_link', name='wrist_2_link', pos=[0, 0.127, 0],
    quat=[1, 0, 0, 0], ipos=[0, 0, 0.1], iquat=[1, 0, 0, 0],
    mass=1.219, inertia=[0.0025599, 0.0025599, 0.0021942],
    joint_axis=[0, 0, 1], joint_range=[-6.28319, 6.28319]
)

# From ur5e.xml
# <body name="wrist_3_link" pos="0 0 0.1">
# <inertial mass="0.1879" pos="0 0.0771683 0" quat="1 0 0 1"
# diaginertia="0.000132134 9.90863e-05 9.90863e-05"/>
# <joint name="wrist_3"/>
# <geom material="linkgray" mesh="wrist3" class="visual"/>
# <geom class="eef_collision" pos="0 0.08 0" quat="1 1 0 0" size="0.04 0.02"/>
# <site name="attachment_site" size="0.01" pos="0 0.1 0" quat="-1 1 1 1" rgba="1 0 0 1" group="1"/>

robot.add_body(
    6, parent='wrist_2_link', name='wrist_3_link', pos=[0, 0, 0.1],
    quat=[1, 0, 0, 0], ipos=[0, 0.0771683, 0], iquat=[1, 0, 0, 1],
    mass=0.1889, inertia=[0.000132134, 9.90863e-05, 9.90863e-05],
    joint_axis=[0, 1, 0], joint_range=[-6.28319, 6.28319]
)

# Normalize quaternions using the rotation library
for body_id, body in robot.body.items():
    body.quat = ram.quat_normalize(body.quat)
    body.iquat = ram.quat_normalize(body.iquat)

print(robot.body[1].parent)
print(robot.body[1].name)
print(robot.body[1].pos)