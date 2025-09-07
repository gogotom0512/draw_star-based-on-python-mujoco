import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from f_k_ur import forward_kinematics
from i_k_ur import inverse_kinematics
from scipy.optimize import fsolve
import utility as ram
import matplotlib.pyplot as plt


xml_path = '../universal_robots_ur5e/scene.xml' #xml file (assumes this is in the same folder as this file)
simend = 30 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

t_init  = 1 #settling time,
f = 1
r = 0.1
def circle(t,x0,y0,r,f):
    x = x0+r*np.cos(2*np.pi*f*t)
    y = y0+r*np.sin(2*np.pi*f*t)
    return x,y

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])
cam.azimuth = -130
cam.elevation = -5
cam.distance =  2
cam.lookat =np.array([ 0.0 , 0.0 , 0.5 ])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

key_qpos = model.key("home").qpos
q = key_qpos.copy()
q = np.array([-1.23, -1.5, 0.5, -1.5708, -1.5708, 0])

#reference
x_ref = 0.5
y_ref = 0.2
z_ref = 0.5
phi_ref = 3.14
theta_ref = 0
psi_ref = 0

opt.frame = mj.mjtFrame.mjFRAME_SITE #or use 3 (check https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjvoption)

time_all = []
x_ref_all = []
y_ref_all = []
z_ref_all = []
x_all = []
y_all = []
z_all = []

while not glfw.window_should_close(window):
    time_prev = data.time

    while data.time - time_prev < 1.0/60.0:

        data.time += 0.02
        data.qpos = q.copy()
        mj.mj_forward(model,data)
        #mj.mj_step(model, data)

        if data.time<t_init:
            #initialize the position
            x0 = x_ref - r
            y0 = y_ref
        else:
            x_ref,y_ref = circle(data.time-t_init,x0,y0,r,f)

        X_ref = np.array([x_ref,y_ref,z_ref,phi_ref,theta_ref,psi_ref])
        q = fsolve(inverse_kinematics,q,args=(X_ref))

        mj_end_eff_pos = data.site('attachment_site').xpos
        mj_end_eff_mat = data.site('attachment_site').xmat

        mj_end_eff_quat = np.zeros(4)
        mj.mju_mat2Quat(mj_end_eff_quat, mj_end_eff_mat)
        #print(mj_end_eff_quat)

        print(f"mj_end_eff_pos = ",mj_end_eff_pos)
        print(f"mj_end_eff_quat = ", mj_end_eff_quat)

        print(f"ref end_eff_pos = ",X_ref[:3])
        ref_euler = ram.euler2quat(np.array([phi_ref,theta_ref,psi_ref]))
        print(f"ref end_eff_quat = ", ref_euler)


        # sol,robot = forward_kinematics(q)
        # py_end_eff_pos = sol.end_eff_pos
        # py_end_eff_rot = sol.end_eff_rot
        # py_end_eff_quat = ram.rotation2quat(py_end_eff_rot)

        # print(f"py_end_eff_pos = ",py_end_eff_pos)
        # print(f"py_end_eff_quat = ", py_end_eff_quat)

        print(' ***** ')

        if data.time>t_init:
            time_all.append(data.time-t_init)
            x_ref_all.append(x_ref)
            y_ref_all.append(y_ref)
            z_ref_all.append(z_ref)
            x_all.append(mj_end_eff_pos[0])
            y_all.append(mj_end_eff_pos[1])
            z_all.append(mj_end_eff_pos[2])

    if data.time>=simend:
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()

# Plot the data
plt.figure()
plt.plot(x_ref_all, y_ref_all, label='ref', color='black', linestyle='--', linewidth=2)
plt.plot(x_all, y_all, label='act', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Trajectory')
plt.grid()
plt.axis('equal')
plt.show()
#plt.show(block=False)  # Non-blocking display
#plt.pause(2)  # Pause for 2 seconds
#plt.close()  # Close the plot
