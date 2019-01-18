# import matplotlib as mpl
# mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from threading import Thread
import time
import xml.etree.ElementTree as xml

from tkinter import TclError

try:
    import openravepy
    from baxter_gym.util_classes.openrave_body import OpenRAVEBody
    from baxter_gym.robot_info.robots import Baxter
    USE_OPENRAVE = True
except:
    USE_OPENRAVE = False

from dm_control import render
from dm_control.mujoco import Physics
from dm_control.viewer import gui
from dm_control.viewer import renderer
from dm_control.viewer import runtime
from dm_control.viewer import user_input
from dm_control.viewer import util
from dm_control.viewer import viewer
from dm_control.viewer import views

from gym import spaces
from gym.core import Env

import baxter_gym
from baxter_gym.util_classes.ik_controller import BaxterIKController
from baxter_gym.util_classes.mjc_xml_utils import *
from baxter_gym.util_classes import transform_utils as T


BASE_VEL_XML = baxter_gym.__path__[0]+'/robot_info/baxter_model.xml'
ENV_XML = baxter_gym.__path__[0]+'/robot_info/current_baxter_env.xml'


MUJOCO_JOINT_ORDER = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint',\
                      'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2', 'left_gripper_l_finger_joint', 'left_gripper_r_finger_joint']
                      

NO_CLOTH = 0
NO_FOLD = 1
ONE_FOLD = 2
TWO_FOLD = 3
WIDTH_GRASP = 4
LENGTH_GRASP = 5
TWO_GRASP = 6
HALF_WIDTH_GRASP = 7
HALF_LENGTH_GRASP = 8
TWIST_FOLD = 9
RIGHT_REACHABLE = 10
LEFT_REACHABLE = 11
IN_RIGHT_GRIPPER = 12
IN_LEFT_GRIPPER = 13
LEFT_FOLD_ON_TOP = 14
RIGHT_FOLD_ON_TOP = 15


BAXTER_GAINS = {
    'left_s0': (5000., 0.01, 2.5),
    'left_s1': (5000., 50., 50.),
    'left_e0': (4000., 15., 1.),
    'left_e1': (1500, 30, 1.),
    'left_w0': (500, 10, 0.01),
    'left_w1': (500, 0.1, 0.01),
    'left_w2': (1000, 0.1, 0.01),
    'left_gripper_l_finger_joint': (1000, 0.1, 0.01),
    'left_gripper_r_finger_joint': (1000, 0.1, 0.01),

    'right_s0': (5000., 0.01, 2.5),
    'right_s1': (5000., 50., 50.),
    'right_e0': (4000., 15., 1.),
    'right_e1': (1500, 30, 1.),
    'right_w0': (500, 10, 0.01),
    'right_w1': (500, 0.1, 0.01),
    'right_w2': (1000, 0.1, 0.01),
    'right_gripper_l_finger_joint': (1000, 0.1, 0.01),
    'right_gripper_r_finger_joint': (1000, 0.1, 0.01),
}

_MAX_FRONTBUFFER_SIZE = 2048
_CAM_WIDTH = 200
_CAM_HEIGHT = 150

GRASP_THRESHOLD = np.array([0.05, 0.05, 0.025]) # np.array([0.01, 0.01, 0.03])
MJC_TIME_DELTA = 0.002
MJC_DELTAS_PER_STEP = int(1. // MJC_TIME_DELTA)
N_CONTACT_LIMIT = 12

START_EE = [0.6, -0.5, 0.7, 0, 0, 1, 0, 0.6, 0.5, 0.7, 0, 0, 1, 0]
CTRL_MODES = ['joint_angle', 'end_effector', 'end_effector_pos', 'discrete_pos']


class BaxterMJCEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'depth'], 'video.frames_per_second': 67}

    def __init__(self, mode='end_effector', obs_include=[], items=[], cloth_info=None, im_dims=(_CAM_WIDTH, _CAM_HEIGHT), max_iter=250, view=False):
        assert mode in CTRL_MODES, 'Env mode must be one of {0}'.format(CTRL_MODES)
        self.ctrl_mode = mode
        self.active = True

        self.cur_time = 0.
        self.prev_time = 0.

        self.use_viewer = view
        self.use_glew = 'MUJOCO_GL' not in os.environ or os.environ['MUJOCO_GL'] != 'osmesa'
        self.obs_include = obs_include
        self._joint_map_cache = {}
        self._ind_cache = {}
        self._cloth_present = cloth_info is not None
        if self._cloth_present:
            self.cloth_width = cloth_info['width']
            self.cloth_length = cloth_info['length']
            self.cloth_sphere_radius = cloth_info['radius']
            self.cloth_spacing = cloth_info['spacing']

        self.im_wid, self.im_height = im_dims
        self.items = items
        self._item_map = {item[0]: item for item in items}
        self._set_obs_info(obs_include)

        self.ctrl_data = {}
        for joint in BAXTER_GAINS:
            self.ctrl_data[joint] = {
                'prev_err': 0.,
                'cp': 0.,
                'cd': 0.,
                'ci': 0.,
            }

        self.ee_ctrl_data = {}
        for joint in BAXTER_GAINS:
            self.ee_ctrl_data[joint] = {
                'prev_err': 0.,
                'cp': 0.,
                'cd': 0.,
                'ci': 0.,
            }

        self._load_model()

        if USE_OPENRAVE:
            env = openravepy.Environment()
            self._ikbody = OpenRAVEBody(env, 'baxter', Baxter())
        else:
            self._ikcontrol = BaxterIKController(lambda: self.get_arm_joint_angles())

        # Start joints with grippers pointing downward
        self.physics.data.qpos[1:8] = self._calc_ik(START_EE[:3], START_EE[3:7], True, False)
        self.physics.data.qpos[10:17] = self._calc_ik(START_EE[7:10], START_EE[10:14], False, False)
        self.physics.forward()

        self.action_inds = {
            ('baxter', 'rArmPose'): np.array(range(7)),
            ('baxter', 'rGripper'): np.array([7]),
            ('baxter', 'lArmPose'): np.array(range(8, 15)),
            ('baxter', 'lGripper'): np.array([15]),
        }

        self._max_iter = max_iter
        self._cur_iter = 0

        if view:
            self._launch_viewer(_CAM_WIDTH, _CAM_HEIGHT)
        else:
            self._viewer = None


    def _load_model(self):
        generate_xml(BASE_VEL_XML, ENV_XML, self.items)
        self.physics = Physics.from_xml_path(ENV_XML)


    def _launch_viewer(self, width, height, title='Main'):
        self._matplot_view_thread = None
        if self.use_glew:
            self._renderer = renderer.NullRenderer()
            self._render_surface = None
            self._viewport = renderer.Viewport(width, height)
            self._window = gui.RenderWindow(width, height, title)
            self._viewer = viewer.Viewer(
                self._viewport, self._window.mouse, self._window.keyboard)
            self._viewer_layout = views.ViewportLayout()
            self._viewer.render()
        else:
            self._viewer = None
            self._matplot_im = None
            self._run_matplot_view()


    def _reload_viewer(self):
        if self._viewer is None or not self.use_glew: return

        if self._render_surface:
          self._render_surface.free()

        if self._renderer:
          self._renderer.release()

        self._render_surface = render.Renderer(
            max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
        self._renderer = renderer.OffScreenRenderer(
            self.physics.model, self._render_surface)
        self._renderer.components += self._viewer_layout
        self._viewer.initialize(
            self.physics, self._renderer, touchpad=False)
        self._viewer.zoom_to_scene()


    def _render_viewer(self, pixels):
        if self.use_glew:
            with self._window._context.make_current() as ctx:
                ctx.call(
                    self._window._update_gui_on_render_thread, self._window._context.window, pixels)
            self._window._mouse.process_events()
            self._window._keyboard.process_events()
        else:
            if self._matplot_im is not None:
                self._matplot_im.set_data(pixels)
                plt.draw()


    def _run_matplot_view(self):
        self._matplot_view_thread = Thread(target=self._launch_matplot_view)
        self._matplot_view_thread.daemon = True
        self._matplot_view_thread.start()


    def _launch_matplot_view(self):
        try:
            self._matplot_im = plt.imshow(self.render(view=False))
            plt.show()
        except TclError:
            print('\nCould not find display to launch viewer (this does not affect the ability to render images)\n')


    def _set_obs_info(self, obs_include):
        self._obs_inds = {}
        self._obs_shape = {}
        ind = 0
        if 'overhead_image' in obs_include or not len(obs_include):
            self._obs_inds['overhead_image'] = (ind, ind+3*self.im_wid*self.im_height)
            self._obs_shape['overhead_image'] = (self.im_height, self.im_wid, 3)
            ind += 3*self.im_wid*self.im_height

        if 'right_image' in obs_include or not len(obs_include):
            self._obs_inds['right_image'] = (ind, ind+3*self.im_wid*self.im_height)
            self._obs_shape['right_image'] = (self.im_height, self.im_wid, 3)
            ind += 3*self.im_wid*self.im_height

        if 'left_image' in obs_include or not len(obs_include):
            self._obs_inds['left_image'] = (ind, ind+3*self.im_wid*self.im_height)
            self._obs_shape['left_image'] = (self.im_height, self.im_wid, 3)
            ind += 3*self.im_wid*self.im_height

        if 'joints' in obs_include or not len(obs_include):
            self._obs_inds['joints'] = (ind, ind+18)
            self._obs_shape['joints'] = (18,)
            ind += 18

        if 'end_effector' in obs_include or not len(obs_include):
            self._obs_inds['end_effector'] = (ind, ind+16)
            self._obs_shape['end_effector'] = (16,)
            ind += 16

        for item, xml, info in self.items:
            if item in obs_include or not len(obs_include):
                self._obs_inds[item] = (ind, ind+3) # Only store 3d Position
                self._obs_shape[item] = (3,)
                ind += 3

        self.dO = ind
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ind,), dtype='float32')
        return ind


    def get_obs(self, obs_include=None):
        obs = np.zeros(self.dO)
        if obs_include is None:
            obs_include = self.obs_include

        if not len(obs_include) or 'overhead_image' in obs_include:
            pixels = self.render(height=self.im_height, width=self.im_wid, camera_id=0, view=False)
            inds = self._obs_inds['overhead_image']
            obs[inds[0]:inds[1]] = pixels.flatten()

        if not len(obs_include) or 'right_image' in obs_include:
            pixels = self.render(height=self.im_height, width=self.im_wid, camera_id=2, view=False)
            inds = self._obs_inds['right_image']
            obs[inds[0]:inds[1]] = pixels.flatten()

        if not len(obs_include) or 'left_image' in obs_include:
            pixels = self.render(height=self.im_height, width=self.im_wid, camera_id=3, view=False)
            inds = self._obs_inds['left_image']
            obs[inds[0]:inds[1]] = pixels.flatten()

        if not len(obs_include) or 'joints' in obs_include:
            jnts = self.get_joint_angles()
            inds = self._obs_inds['joints']
            obs[inds[0]:inds[1]] = jnts

        if not len(obs_include) or 'end_effector' in obs_include:
            grip_jnts = self.get_gripper_joint_angles()
            inds = self._obs_inds['end_effector']
            obs[inds[0]:inds[1]] = np.r_[self.get_right_ee_pos(), 
                                         self.get_right_ee_rot(),
                                         grip_jnts[0],
                                         self.get_left_ee_pos(),
                                         self.get_left_ee_rot(),
                                         grip_jnts[1]]

        for item in self.items:
            if not len(obs_include) or item[0] in obs_include:
                inds = self._obs_inds[item[0]]
                obs[inds[0]:inds[1]] = self.get_item_pose(item[0])

        return np.array(obs)


    def get_obs_types(self):
        return self._obs_inds.keys()


    def get_obs_inds(self, obs_type):
        if obs_type not in self._obs_inds:
            raise KeyError('{0} is not a valid observation for this environment. Valid options: {1}'.format(obs_type, self.get_obs_types()))
        return self._obs_inds[obs_type]


    def get_obs_shape(self, obs_type):
        if obs_type not in self._obs_inds:
            raise KeyError('{0} is not a valid observation for this environment. Valid options: {1}'.format(obs_type, self.get_obs_types()))
        return self._obs_shape[obs_type]


    def get_obs_data(self, obs, obs_type):
        if obs_type not in self._obs_inds:
            raise KeyError('{0} is not a valid observation for this environment. Valid options: {1}'.format(obs_type, self.get_obs_types()))
        return obs[self._obs_inds[obs_type]].reshape(self._obs_shape[obs_type])


    def get_arm_section_inds(self, section_name):
        inds = self.get_obs_inds('joints')
        if section_name == 'lArmPose':
            return inds[9:16]
        if section_name == 'lGripper':
            return inds[16:]
        if section_name == 'rArmPose':
            return inds[:7]
        if section_name == 'rGripper':
            return inds[7:8]


    def get_left_ee_pos(self, mujoco_frame=True):
        model = self.physics.model
        ll_gripper_ind = model.name2id('left_gripper_l_finger_tip', 'body')
        lr_gripper_ind = model.name2id('left_gripper_r_finger_tip', 'body')
        pos = (self.physics.data.xpos[ll_gripper_ind] + self.physics.data.xpos[lr_gripper_ind]) / 2
        if not mujoco_frame:
            pos[2] -= MUJOCO_MODEL_Z_OFFSET
        return pos 


    def get_right_ee_pos(self, mujoco_frame=True):
        model = self.physics.model
        rr_gripper_ind = model.name2id('right_gripper_r_finger_tip', 'body')
        rl_gripper_ind = model.name2id('right_gripper_l_finger_tip', 'body')
        pos = (self.physics.data.xpos[rr_gripper_ind] + self.physics.data.xpos[rl_gripper_ind]) / 2
        if not mujoco_frame:
            pos[2] -= MUJOCO_MODEL_Z_OFFSET
        return pos


    def get_left_ee_rot(self):
        model = self.physics.model
        l_gripper_ind = model.name2id('left_gripper_base', 'body')
        return self.physics.data.xquat[l_gripper_ind].copy()


    def get_right_ee_rot(self):
        model = self.physics.model
        r_gripper_ind = model.name2id('right_gripper_base', 'body')
        return self.physics.data.xquat[r_gripper_ind].copy()


    def get_item_pose(self, name, mujoco_frame=True):
        model = self.physics.model
        if name in self._ind_cache:
            item_ind = self._ind_cache[name]
        else:
            try:
                item_ind = model.name2id(name, 'body')
            except:
                item_ind = -1
            self._ind_cache[name] = item_ind
        pos = self.physics.data.xpos[item_ind].copy()
        if not mujoco_frame:
            pos[2] -= MUJOCO_MODEL_Z_OFFSET
        return pos


    def get_item_rot(self, name, convert_to_euler=False):
        model = self.physics.model
        item_ind = model.name2id(name, 'body')
        rot = self.physics.data.xquat[item_ind].copy()
        if convert_to_euler:
            rot = tf.euler_from_quaternion(rot)
        return rot


    def get_pos_from_label(self, label):
        if label in self._item_map:
            return self.get_item_pose(label)
        return None


    def get_joint_angles(self):
        return self.physics.data.qpos[1:19].copy()


    def get_arm_joint_angles(self):
        inds = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
        return self.physics.data.qpos[inds]


    def get_gripper_joint_angles(self):
        inds = [8, 17]
        return self.physics.data.qpos[inds]


    def _get_joints(self, act_index):
        if act_index in self._joint_map_cache:
            return self._joint_map_cache[act_index]

        res = []
        for name, attr in self.action_inds:
            inds = self.action_inds[name, attr]
            # Actions have a single gripper command, but MUJOCO uses two gripper joints
            if act_index in inds:
                if attr == 'lGripper':
                    res = [('left_gripper_l_finger_joint', 1), ('left_gripper_r_finger_joint', -1)]
                elif attr == 'rGripper':
                    res = [('right_gripper_r_finger_joint', 1), ('right_gripper_l_finger_joint', -1)]
                elif attr == 'lArmPose':
                    arm_ind = inds.tolist().index(act_index)
                    res = [(MUJOCO_JOINT_ORDER[9+arm_ind], 1)]
                elif attr == 'rArmPose':
                    arm_ind = inds.tolist().index(act_index)
                    res = [(MUJOCO_JOINT_ORDER[arm_ind], 1)]

        self._joint_map_cache[act_index] = res
        return res


    def get_action_meanings(self):
        # For discrete action mode
        return ['NOOP', 'RIGHT_EE_FORWARD', 'RIGHT_EE_BACK', 'RIGHT_EE_LEFT', 'RIGHT_EE_RIGHT',
                'RIGHT_EE_UP', 'RIGHT_EE_DOWN', 'RIGHT_EE_OPEN', 'RIGHT_EE_CLOSE',
                'LEFT_EE_FORWARD', 'LEFT_EE_BACK', 'LEFT_EE_LEFT', 'LEFT_EE_RIGHT',
                'LEFT_EE_UP', 'LEFT_EE_DOWN', 'LEFT_EE_OPEN', 'LEFT_EE_CLOSE']


    def move_right_gripper_forward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[0] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_backward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[0] = -0.01
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_left(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[1] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_right(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[1] = -0.01
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_up(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[2] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_down(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[2] = -0.01
        return self.step(act, mode='end_effector_pos')

    def open_right_gripper(self):
        act = np.zeros(8)
        act[3] = 0.02
        return self.step(act, mode='end_effector_pos')


    def close_right_gripper(self):
        act = np.zeros(8)
        act[3] = 0
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_forward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[4] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_backward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[4] = -0.01
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_left(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[5] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_right(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[5] = -0.01
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_up(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[6] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_down(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[6] = -0.01
        return self.step(act, mode='end_effector_pos')

    def open_left_gripper(self):
        act = np.zeros(8)
        act[7] = 0.02
        return self.step(act, mode='end_effector_pos')


    def close_left_gripper(self):
        act = np.zeros(8)
        act[7] = 0
        return self.step(act, mode='end_effector_pos')


    def _step_joint(self, joint, error):
        ctrl_data = self.ctrl_data[joint]
        gains = BAXTER_GAINS[joint]
        dt = MJC_TIME_DELTA
        de = error - ctrl_data['prev_err']
        ctrl_data['cp'] = error
        ctrl_data['cd'] = de / dt
        ctrl_data['ci'] += error * dt
        ctrl_data['prev_err'] = error
        return gains[0] * ctrl_data['cp'] + \
               gains[1] * ctrl_data['cd'] + \
               gains[2] * ctrl_data['ci']


    def _clip_joint_angles(self, r_jnts, r_grip, l_jnts, l_grip):
        DOF_limits = self._ikbody.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9]+0.001, DOF_limits[1][2:9]-0.001)
        right_DOF_limits = (DOF_limits[0][10:17]+0.001, DOF_limits[1][10:17]-0.001)

        if r_grip[0] < 0:
            r_grip[0] = 0
        if r_grip[0] > 0.02:
            r_grip[0] = 0.02
        if l_grip[0] < 0:
            l_grip[0] = 0
        if l_grip[0] > 0.02:
            l_grip[0] = 0.02

        for i in range(7):
            if l_jnts[i] < left_DOF_limits[0][i]:
                l_jnts[i] = left_DOF_limits[0][i]
            if l_jnts[i] > left_DOF_limits[1][i]:
                l_jnts[i] = left_DOF_limits[1][i]
            if r_jnts[i] < right_DOF_limits[0][i]:
                r_jnts[i] = right_DOF_limits[0][i]
            if r_jnts[i] > right_DOF_limits[1][i]:
                r_jnts[i] = right_DOF_limits[1][i]


    def _calc_ik(self, pos, quat, use_right=True, check_limits=True):
        arm_jnts = self.get_arm_joint_angles()
        grip_jnts = self.get_gripper_joint_angles()
        if USE_OPENRAVE:
            self._clip_joint_angles(arm_jnts[:7], grip_jnts[:1], arm_jnts[7:], grip_jnts[1:])

            dof_map = {
                'rArmPose': arm_jnts[:7],
                'rGripper': grip_jnts[0],
                'lArmPose': arm_jnts[7:],
                'lGripper': grip_jnts[1],
            }

            manip_name = 'right_arm' if use_right else 'left_arm'
            trans = np.zeros((4, 4))
            trans[:3, :3] = openravepy.matrixFromQuat(quat)[:3,:3]
            trans[:3, 3] = pos + np.array([0.07, 0, 0])
            trans[3, 3] = 1

            jnt_cmd = self._ikbody.get_close_ik_solution(manip_name, trans, dof_map)

        else:
            cmd = {'dpos': pos+np.array([0,0,MUJOCO_MODEL_Z_OFFSET]), 'rotation': [quat[1], quat[2], quat[3], quat[0]]}
            jnt_cmd = self._ikcontrol.joint_positions_for_eef_command(cmd, use_right)

        if use_right:
            if jnt_cmd is None or (check_limits and np.any(np.abs(jnt_cmd - arm_jnts[:7]) > 0.3)):
                print('Cannot complete action; ik will cause unstable control')
                return arm_jnts[:7]
        else:
            if jnt_cmd is None or (check_limits and np.any(np.abs(jnt_cmd - arm_jnts[7:]) > 0.3)):
                print('Cannot complete action; ik will cause unstable control')
                return arm_jnts[7:]

        return jnt_cmd


    def _check_ik(self, pos, quat, use_right=True):
        if USE_OPENRAVE:
            arm_jnts = self.get_arm_joint_angles()
            grip_jnts = self.get_gripper_joint_angles()
            self._clip_joint_angles(arm_jnts[:7], grip_jnts[:1], arm_jnts[7:], grip_jnts[1:])

            dof_map = {
                'rArmPose': arm_jnts[:7],
                'rGripper': grip_jnts[0],
                'lArmPose': arm_jnts[7:],
                'lGripper': grip_jnts[1],
            }

            manip_name = 'right_arm' if use_right else 'left_arm'
            trans = np.zeros((4, 4))
            trans[:3, :3] = openravepy.matrixFromQuat(quat)[:3,:3]
            trans[:3, 3] = pos + np.array([0.07, 0, 0])
            trans[3, 3] = 1
            jnt_cmd = self._ikbody.get_close_ik_solution(manip_name, trans, dof_map)
        else:
            cmd = {'dpos': pos+np.array([0,0,MUJOCO_MODEL_Z_OFFSET]), 'rotation': [quat[1], quat[2], quat[3], quat[0]]}
            jnt_cmd = self._ikcontrol.joint_positions_for_eef_command(cmd, use_right)

        return jnt_cmd is not None


    def step(self, action, mode=None, obs_include=None, debug=False):
        if mode is None:
            mode = self.ctrl_mode

        cmd = np.zeros((18))
        abs_cmd = np.zeros((18))

        r_grip = 0
        l_grip = 0

        if mode == 'joint_angle':
            for i in range(len(action)):
                jnts = self._get_joints(i)
                for jnt in jnts:
                    cmd_angle = jnt[1] * action[i]
                    ind = MUJOCO_JOINT_ORDER.index(jnt[0])
                    abs_cmd[ind] = cmd_angle
            r_grip = action[7]
            l_grip = action[15]

        elif mode == 'end_effector':
            # Action Order: ee_right_pos, ee_right_quat, ee_right_grip, ee_left_pos, ee_left_quat, ee_left_grip
            cur_right_ee_pos = self.get_right_ee_pos()
            cur_right_ee_rot = self.get_right_ee_rot()
            cur_left_ee_pos = self.get_left_ee_pos()
            cur_left_ee_rot = self.get_left_ee_rot()

            target_right_ee_pos = cur_right_ee_pos + action[:3]
            target_right_ee_pos[2] -= MUJOCO_MODEL_Z_OFFSET
            target_right_ee_rot = action[3:7] # cur_right_ee_rot + action[3:7]
            target_left_ee_pos = cur_left_ee_pos + action[8:11]
            target_left_ee_pos[2] -= MUJOCO_MODEL_Z_OFFSET
            target_left_ee_rot = action[11:15] # cur_left_ee_rot + action[11:15]

            # target_right_ee_rot /= np.linalg.norm(target_right_ee_rot)
            # target_left_ee_rot /= np.linalg.norm(target_left_ee_rot)

            right_cmd = self._calc_ik(target_right_ee_pos, 
                                      target_right_ee_rot, 
                                      use_right=True)

            left_cmd = self._calc_ik(target_left_ee_pos, 
                                     target_left_ee_rot, 
                                     use_right=False)

            abs_cmd[:7] = right_cmd
            abs_cmd[9:16] = left_cmd
            r_grip = action[7]
            l_grip = action[15]

        elif mode == 'end_effector_pos':
            # Action Order: ee_right_pos, ee_right_quat, ee_right_grip, ee_left_pos, ee_left_quat, ee_left_grip
            cur_right_ee_pos = self.get_right_ee_pos()
            cur_left_ee_pos = self.get_left_ee_pos()

            target_right_ee_pos = cur_right_ee_pos + action[:3]
            target_right_ee_pos[2] -= MUJOCO_MODEL_Z_OFFSET
            target_right_ee_rot = START_EE[3:7]
            target_left_ee_pos = cur_left_ee_pos + action[4:7]
            target_left_ee_pos[2] -= MUJOCO_MODEL_Z_OFFSET
            target_left_ee_rot = START_EE[10:14]

            right_cmd = self._calc_ik(target_right_ee_pos, 
                                      target_right_ee_rot, 
                                      use_right=True)

            left_cmd = self._calc_ik(target_left_ee_pos, 
                                     target_left_ee_rot, 
                                     use_right=False)

            abs_cmd[:7] = right_cmd
            abs_cmd[9:16] = left_cmd
            r_grip = action[3]
            l_grip = action[7]

        elif mode == 'discrete_pos':
            if action == 1: return self.move_right_gripper_forward()
            if action == 2: return self.move_right_gripper_backward()
            if action == 3: return self.move_right_gripper_left()
            if action == 4: return self.move_right_gripper_right()
            if action == 5: return self.move_right_gripper_up()
            if action == 6: return self.move_right_gripper_down()
            if action == 7: return self.open_right_gripper()
            if action == 8: return self.close_right_gripper()

            if action == 9: return self.move_left_gripper_forward()
            if action == 10: return self.move_left_gripper_backward()
            if action == 11: return self.move_left_gripper_left()
            if action == 12: return self.move_left_gripper_right()
            if action == 13: return self.move_left_gripper_up()
            if action == 14: return self.move_left_gripper_down()
            if action == 15: return self.open_left_gripper()
            if action == 16: return self.close_left_gripper()
            return self.get_obs(), \
                   self.compute_reward(), \
                   False, \
                   {}

        for t in range(MJC_DELTAS_PER_STEP / 4):
            error = abs_cmd - self.physics.data.qpos[1:19]
            cmd = 7e1 * error
            cmd[7] = 20 if r_grip > 0.0175 else -75
            cmd[8] = -cmd[7]
            cmd[16] = 20 if l_grip > 0.0175 else -75
            cmd[17] = -cmd[16]
            self.physics.set_control(cmd)
            self.physics.step()

        return self.get_obs(obs_include=obs_include), \
               self.compute_reward(), \
               self.is_done(), \
               {}


    def compute_reward(self):
        return 0


    def is_done(self):
        return self._cur_iter >= self._max_iter


    def render(self, mode='rgb_array', height=0, width=0, camera_id=0,
               overlays=(), depth=False, scene_option=None, view=True):
        # Make friendly with dm_control or gym interface
        depth = depth or mode == 'depth_array'
        view = view or mode == 'human'
        if height == 0: height = self.im_height
        if width == 0: width = self.im_wid

        pixels = self.physics.render(height, width, camera_id, overlays, depth, scene_option)
        if view and self.use_viewer:
            self._render_viewer(pixels)

        return pixels


    def reset(self):
        self._cur_iter = 0
        self.physics.reset()
        self._reload_viewer()
        self.ctrl_data = {}
        self.cur_time = 0.
        self.prev_time = 0.
        for joint in BAXTER_GAINS:
            self.ctrl_data[joint] = {
                'prev_err': 0.,
                'cp': 0.,
                'cd': 0.,
                'ci': 0.,
            }
        return self.get_obs()


    @classmethod
    def init_from_plan(cls, plan, view=True):
        items = []
        for p in plan.params.valuyes():
            if p.is_symbol(): continue
            param_xml = get_param_xml(p)
            if param_xml is not None:
                items.append(param_xml)
        return cls.__init__(view, items)


    def sim_from_plan(self, plan, t):
        model  = self.physics.model
        xpos = model.body_pos.copy()
        xquat = model.body_quat.copy()
        param = plan.params.values()

        for param_name in plan.params:
            param = plan.params[param_name]
            if param.is_symbol(): continue
            if param._type != 'Robot':
                if param.name in self._ind_cache:
                    param_ind = self._ind_cache[param.name]
                else:
                    try:
                        param_ind = model.name2id(param.name, 'body')
                    except:
                        param_ind = -1
                    self._ind_cache[param.name] = -1
                if param_ind == -1: continue

                pos = param.pose[:, t]
                xpos[param_ind] = pos + np.array([0, 0, MUJOCO_MODEL_Z_OFFSET])
                if hasattr(param, 'rotation'):
                    rot = param.rotation[:, t]
                    mat = OpenRAVEBody.transform_from_obj_pose([0,0,0], rot)[:3,:3]
                    xquat[param_ind] = openravepy.quatFromRotationMatrix(mat)

        self.physics.data.xpos[:] = xpos[:]
        self.physics.data.xquat[:] = xquat[:]
        model.body_pos[:] = xpos[:]
        model.body_quat[:] = xquat[:]

        baxter = plan.params['baxter']
        self.physics.data.qpos[1:8] = baxter.rArmPose[:, t]
        self.physics.data.qpos[8] = baxter.rGripper[:, t]
        self.physics.data.qpos[9] = -baxter.rGripper[:, t]
        self.physics.data.qpos[10:17] = baxter.lArmPose[:, t]
        self.physics.data.qpos[17] = baxter.lGripper[:, t]
        self.physics.data.qpos[18] = -baxter.lGripper[:, t]

        self.physics.forward()


    def mp_state_from_sim(self, plan):
        X = np.zeros(plan.symbolic_bound)
        for param_name, attr_name in plan.state_inds:
            inds = plan.state_inds[param_name, attr_name]
            if param_name in plan.params:
                param = plan.params[param_name]
                if param_name == 'baxter':
                    pass
                elif not param.is_symbol():
                    if attr_name == 'pose':
                        X[inds] = self.get_item_pose(param_name)
                    elif attr_name == 'rotation':
                        X[inds] = self.get_item_rot(param_name, convert_to_euler=True)




    def jnt_ctrl_from_plan(self, plan, t):
        baxter = plan.params['baxter']
        lArmPose = baxter.lArmPose[:, t]
        lGripper = baxter.lGripper[:, t]
        rArmPose = baxter.rArmPose[:, t]
        rGripper = baxter.rGripper[:, t]
        ctrl = np.r_[rArmPose, rGripper, -rGripper, lArmPose, lGripper, -lGripper]
        return self.step(joint_angles=ctrl)


    def run_plan(self, plan):
        self.reset()
        obs = []
        for t in range(plan.horizon):
            obs.append(self.jnt_ctrl_from_plan(plan, t))

        return obs


    def close(self):
        self.active = False
        if self._viewer is not None and self.use_glew:
            self._viewer.close()
            self._viewer = None
        self.physics.free()


    def seed(self, seed=None):
        pass


    def list_joint_info(self):
        for i in range(self.physics.model.njnt):
            print('\n')
            print('Jnt ', i, ':', self.physics.model.id2name(i, 'joint'))
            print('Axis :', self.physics.model.jnt_axis[i])
            print('Dof adr :', self.physics.model.jnt_dofadr[i])
            body_id = self.physics.model.jnt_bodyid[i]
            print('Body :', self.physics.model.id2name(body_id, 'body'))
            print('Parent body :', self.physics.model.id2name(self.physics.model.body_parentid[body_id], 'body'))
