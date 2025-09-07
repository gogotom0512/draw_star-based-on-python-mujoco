import numpy as np
import utility as ram
from f_k_ur import forward_kinematics


def inverse_kinematics(q, X_ref):
    # unpack X_ref
    x_ref = X_ref[0]
    y_ref = X_ref[1]
    z_ref = X_ref[2]
    roll_ref = X_ref[3]
    pitch_ref = X_ref[4]
    yaw_ref = X_ref[5]

    quat_ref = ram.euler2quat(np.array([roll_ref, pitch_ref, yaw_ref]))
    R_ref = ram.euler2rotation(np.array([roll_ref, pitch_ref, yaw_ref]))

    sol, robot = forward_kinematics(q)
    py_end_eff_pos = sol.end_eff_pos
    py_end_eff_rot = sol.end_eff_rot
    py_end_eff_quat = ram.rotation2quat(py_end_eff_rot)
    py_end_eff_euler = ram.rotation2euler(py_end_eff_rot)

    x = py_end_eff_pos[0]
    y = py_end_eff_pos[1]
    z = py_end_eff_pos[2]
    phi = py_end_eff_euler[0]
    theta = py_end_eff_euler[1]
    psi = py_end_eff_euler[2]

    # R*R_ref^T = eye(3)
    RRt = py_end_eff_rot @ R_ref.T

    dquat = np.zeros(4)
    for i in range(0, 4):
        dquat[i] = py_end_eff_quat[i] - quat_ref[i]

    # provide what needs to be zerod and return it

    # return x-x_ref,y-y_ref,z-z_ref,phi-roll_ref,theta-pitch_ref,psi-yaw_ref
    # return x-x_ref,y-y_ref,z-z_ref,dquat[1],dquat[2],dquat[3]
    return x - x_ref, y - y_ref, z - z_ref, RRt[0][0] - 1, RRt[1][1] - 1, RRt[2][2] - 1