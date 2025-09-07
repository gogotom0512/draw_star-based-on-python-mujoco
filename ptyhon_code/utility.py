import numpy as np

def rotation(angle, axis):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:  # Rotation around x-axis
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 1:  # Rotation around y-axis
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 2:  # Rotation around z-axis
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 0 (x), 1 (y), or 2 (z).")

import numpy as np

def quat2rotation(q):

    # Extract quaternion components
    q0, q1, q2, q3 = q

    # Compute the rotation matrix using the formula
    R = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2]
    ])

    return R
# scipy
def rotation2quat(R):
    # Ensure R is a 3x3 matrix
    assert R.shape == (3, 3), "Input must be a 3x3 rotation matrix."

    # Calculate the magnitudes of q0, q1, q2, q3
    q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    q1 = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
    q2 = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) / 2
    q3 = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) / 2

    # Determine which magnitude is largest and compute the quaternion components
    if q0 >= max(q1, q2, q3):  # q0 is largest
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    elif q1 >= max(q0, q2, q3):  # q1 is largest
        q0 = (R[2, 1] - R[1, 2]) / (4 * q1)
        q2 = (R[1, 0] + R[0, 1]) / (4 * q1)
        q3 = (R[0, 2] + R[2, 0]) / (4 * q1)
    elif q2 >= max(q0, q1, q3):  # q2 is largest
        q0 = (R[0, 2] - R[2, 0]) / (4 * q2)
        q1 = (R[1, 0] + R[0, 1]) / (4 * q2)
        q3 = (R[2, 1] + R[1, 2]) / (4 * q2)
    else:  # q3 is largest
        q0 = (R[1, 0] - R[0, 1]) / (4 * q3)
        q1 = (R[0, 2] + R[2, 0]) / (4 * q3)
        q2 = (R[2, 1] + R[1, 2]) / (4 * q3)

    return np.array([q0, q1, q2, q3])

def quat2axisangle(quat):
    #q0 = cos(angle/2)
    #[qx,qy,qz] = sin(angle/2)*axis

    #angle = 2*acos(q0)
    #axis = (1/sin(angle/2))*[qx,qy,qz]

    q0 = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    angle = 2*np.arccos(q0)
    sin_half_theta = np.sin(angle/2);

    if (sin_half_theta < 1e-6): #avoid division by zeros
        axis = np.array([1,0,0]) #this is just an arbitrary choice
    else:
        axis = (1/sin_half_theta)*np.array([qx,qy,qz])

    return axis,angle


def euler2rotation(euler):
    Rx = rotation(euler[0], 0)
    Ry = rotation(euler[1], 1)
    Rz = rotation(euler[2], 2)
    R = Rx@Ry
    R = R@Rz
    return R

def rotation2euler(R):
    r13 = R[0,2]
    theta = np.arcsin(r13); #sin(theta) = r13
    cos_theta = np.cos(theta)

    r12 = R[0,1]
    psi = np.arcsin(-r12/cos_theta) #sin(psi)*cos(theta) = -r12

    r23 = R[1,2]
    phi = np.arcsin(-r23/cos_theta) #sin(phi)*cos(theta) = -r23

    return np.array([phi,theta,psi])

# def mat2euler(mat):
#     """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
#     # For testing whether a number is close to zero
#     _FLOAT_EPS = np.finfo(np.float64).eps
#     _EPS4 = _FLOAT_EPS * 4.0
#
#     mat = np.asarray(mat, dtype=np.float64)
#     assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)
#
#     cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
#     condition = cy > _EPS4
#     euler = np.empty(mat.shape[:-1], dtype=np.float64)
#     euler[..., 2] = np.where(condition,
#                              -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
#                              -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
#     euler[..., 1] = np.where(condition,
#                              -np.arctan2(-mat[..., 0, 2], cy),
#                              -np.arctan2(-mat[..., 0, 2], cy))
#     euler[..., 0] = np.where(condition,
#                              -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
#                              0.0)
#     return euler


def quat2euler(q):
    R = quat2rotation(q)
    euler = rotation2euler(R)

    return euler

def euler2quat(euler):
    R = euler2rotation(euler)
    quat = rotation2quat(R)
    return quat


def quat_product(q,p):
    q0 = q[0]
    p0 = p[0]
    p_vec = p[1:4].copy()
    q_vec = q[1:4].copy()
    qp0 = q0*p0-np.dot(p_vec,q_vec)
    qp_vec = q0*p_vec + p0*q_vec + np.cross(q_vec,p_vec)
    qp = np.array([qp0,qp_vec[0],qp_vec[1],qp_vec[2]])
    return qp

def quat_normalize(q):
    assert q.shape[-1] == 4
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / norm  # Normalize each quaternion

def vec2skew(v):
    # Ensure v is a 3-element vector
    assert len(v) == 3, "Input must be a 3-element vector"

    # Create the skew-symmetric matrix
    V = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    return V

# Taken from: https://github.com/openai/mujoco-worldgen/blob/master/mujoco_worldgen/util/rotation.py
def mat2quat(mat):
    """ Convert Rotation Matrix to Quaternion.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q
