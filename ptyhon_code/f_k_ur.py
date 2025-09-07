from robot_data import robot
import utility as ram
import numpy as np

from types import SimpleNamespace

def forward_kinematics(q):

    #Compute H_local
    for i in range(1,len(robot.body)+1): #goes from 1 to 6 (6 bodies)
        quat = robot.body[i].quat
        joint_axis = robot.body[i].joint_axis
        axis_id = np.argmax(np.abs(joint_axis)) #get the joint axis, 0, 1, 2
        angle = q[i-1]  # body_id goes from 1 to 6 but q goes from 0 to 5.

        R_q = ram.rotation(angle, axis_id)
        robot.body[i].R_local= ram.quat2rotation(quat) @ R_q
        robot.body[i].o_local = robot.body[i].pos

        robot.body[i].H_local = np.block([
        [robot.body[i].R_local, robot.body[i].o_local.reshape(-1, 1)], #does 3x1
        [np.zeros((1, 3)), 1]
        ])

    #Compute H_global
    base_quat = robot.params.base_quat
    R_base = ram.quat2rotation(base_quat)

    H_base = np.block([
        [R_base, np.zeros((3, 1))],
        [np.zeros((1, 3)), 1]
    ])

    temp = H_base
    for i in range(1,len(robot.body)+1):
        # Compute global transformation matrix
        robot.body[i].H_global = temp @ robot.body[i].H_local
        # Update temp for the next iteration
        temp = robot.body[i].H_global

    #Compute the position of the end-effector
    end_eff_pos_local = robot.params.end_eff_pos_local
    end_eff_quat_local = robot.params.end_eff_quat_local
    R_end_eff = ram.quat2rotation(end_eff_quat_local )

    #end_eff stuff
    end_eff_pos = robot.body[6].H_global @ np.append(end_eff_pos_local, 1)  # Homogeneous coordinates
    end_eff_pos = end_eff_pos[:3]  # Extract translation

    # Extract the rotation matrix from the global transformation matrix and R_end_eff
    end_eff_rot = robot.body[6].H_global[:3, :3] @ R_end_eff


    #Define sol
    sol = SimpleNamespace(
        end_eff_pos=end_eff_pos,
        end_eff_rot=end_eff_rot,
        )

    return sol,robot