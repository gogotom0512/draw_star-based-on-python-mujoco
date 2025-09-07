import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from f_k_ur import forward_kinematics
from i_k_ur import inverse_kinematics
from scipy.optimize import fsolve
import utility as ram
import matplotlib.pyplot as plt

xml_path = '../universal_robots_ur5e/scene.xml'  # xml file (assumes this is in the same folder as this file)
simend = 30  # simulation time
print_camera_config = 0  # set to 1 to print camera config
# this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

t_init = 2  # settling time, increased slightly for star
f = 0.2  # Frequency, drawing one star every 5 seconds
r = 0.15  # Radius


def pentagram(t, x0, y0, r, f):
    period = 1.0 / f
    t_mod = t % period
    angles = np.array([np.pi / 2 + i * 2 * np.pi / 5 for i in range(5)])
    vertices = np.array([[x0 + r * np.cos(angle), y0 + r * np.sin(angle)] for angle in angles])
    path_indices = [0, 2, 4, 1, 3, 0]
    segment_duration = period / 5.0
    segment_index = int(t_mod / segment_duration)
    if segment_index >= 5:
        segment_index = 4
    start_node = path_indices[segment_index]
    end_node = path_indices[segment_index + 1]
    p1 = vertices[start_node]
    p2 = vertices[end_node]
    t_segment = (t_mod % segment_duration) / segment_duration
    x = p1[0] * (1 - t_segment) + p2[0] * t_segment
    y = p1[1] * (1 - t_segment) + p2[1] * t_segment
    return x, y


def init_controller(model, data):
    pass


def controller(model, data):
    pass


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)


def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx, dy = xpos - lastx, ypos - lasty
    lastx, lasty = xpos, ypos
    if not (button_left or button_middle or button_right):
        return
    width, height = glfw.get_window_size(window)
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)


dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = -130
cam.elevation = -5
cam.distance = 2
cam.lookat = np.array([0.0, 0.0, 0.5])

init_controller(model, data)
mj.set_mjcb_control(controller)

key_qpos = model.key("home").qpos
q = key_qpos.copy()
q = np.array([-1.23, -1.5, 0.5, -1.5708, -1.5708, 0])

x_ref, y_ref, z_ref = 0.5, 0.2, 0.5
phi_ref, theta_ref, psi_ref = 3.14, 0, 0

opt.frame = mj.mjtFrame.mjFRAME_SITE

time_all, x_ref_all, y_ref_all, z_ref_all = [], [], [], []
x_all, y_all, z_all = [], [], []

x0, y0 = x_ref, y_ref

# List to store the trajectory points for visualization in MuJoCo
trajectory_points = []

while not glfw.window_should_close(window):
    time_prev = data.time
    while data.time - time_prev < 1.0 / 60.0:
        data.time += 0.02
        data.qpos = q.copy()
        mj.mj_forward(model, data)

        current_x_ref, current_y_ref = x_ref, y_ref
        if data.time < t_init:
            start_x, start_y = pentagram(0, x0, y0, r, f)
            current_x_ref, current_y_ref = start_x, start_y
        else:
            current_x_ref, current_y_ref = pentagram(data.time - t_init, x0, y0, r, f)

        X_ref = np.array([current_x_ref, current_y_ref, z_ref, phi_ref, theta_ref, psi_ref])
        q = fsolve(inverse_kinematics, q, args=(X_ref))

        mj_end_eff_pos = data.site('attachment_site').xpos

        # Append the current position to the trajectory list
        if data.time > t_init:
            trajectory_points.append(mj_end_eff_pos.copy())
            # Optional: Limit the trail length
            if len(trajectory_points) > 1000:
                trajectory_points.pop(0)

        if data.time > t_init:
            time_all.append(data.time - t_init)
            x_ref_all.append(current_x_ref)
            y_ref_all.append(current_y_ref)
            z_ref_all.append(z_ref)
            x_all.append(mj_end_eff_pos[0])
            y_all.append(mj_end_eff_pos[1])
            z_all.append(mj_end_eff_pos[2])

    if data.time >= simend:
        break

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    if print_camera_config == 1:
        print(f"cam.azimuth = {cam.azimuth}; cam.elevation = {cam.elevation}; cam.distance = {cam.distance}")
        print(f"cam.lookat = np.array([{cam.lookat[0]}, {cam.lookat[1]}, {cam.lookat[2]}])")

    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)

    # --- Add trajectory visualization geoms ---
    for point in trajectory_points:
        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(scene.geoms[scene.ngeom],
                            type=mj.mjtGeom.mjGEOM_SPHERE,
                            size=np.array([0.005, 0, 0]),
                            pos=point,
                            mat=np.identity(3).flatten(),
                            rgba=np.array([1.0, 0.0, 0.0, 0.7]))
            scene.ngeom += 1

    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()

plt.figure(figsize=(8, 8))
plt.plot(x_ref_all, y_ref_all, label='Reference Trajectory', color='black', linestyle='--', linewidth=2)
plt.plot(x_all, y_all, label='Actual Trajectory', color='red', alpha=0.8)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.title('Pentagram Trajectory Tracking')
plt.grid(True)
plt.axis('equal')
plt.show()