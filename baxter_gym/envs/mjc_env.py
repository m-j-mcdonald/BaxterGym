import matplotlib.pyplot as plt
import numpy as np
import os
from threading import Thread
import time
from tkinter import TclError
import traceback
import sys
import xml.etree.ElementTree as xml


try:
    from dm_control import render
except:
    from dm_control import _render as render
from dm_control.mujoco import Physics
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.rl.control import PhysicsError
from dm_control.viewer import gui
from dm_control.viewer import renderer
from dm_control.viewer import runtime
from dm_control.viewer import user_input
from dm_control.viewer import util
from dm_control.viewer import viewer
from dm_control.viewer import views

from gym import spaces
from gym.core import Env

from baxter_gym.util_classes.mjc_xml_utils import *
from baxter_gym.util_classes import transform_utils as T


_MAX_FRONTBUFFER_SIZE = 2048
_CAM_WIDTH = 200
_CAM_HEIGHT = 150

CTRL_MODES = ['joint_angle', 'end_effector', 'end_effector_pos', 'discrete_pos']

class MJCEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'depth'], 'video.frames_per_second': 67}

    def __init__(self, mode='end_effector', obs_include=[], items=[], include_files=[], include_items=[], im_dims=(_CAM_WIDTH, _CAM_HEIGHT), sim_freq=25, timestep=0.002, max_iter=250, view=False):
        assert mode in CTRL_MODES, 'Env mode must be one of {0}'.format(CTRL_MODES)
        self.ctrl_mode = mode
        self.active = True

        self.cur_time = 0.
        self.prev_time = 0.
        self.timestep = timestep
        self.sim_freq = sim_freq

        self.use_viewer = view
        self.use_glew = 'MUJOCO_GL' not in os.environ or os.environ['MUJOCO_GL'] == 'glfw'
        self.obs_include = obs_include
        self._joint_map_cache = {}
        self._ind_cache = {}

        self.im_wid, self.im_height = im_dims
        self.items = items
        self._item_map = {item[0]: item for item in items}
        self.include_files = include_files
        self.include_items = include_items
        self._set_obs_info(obs_include)

        self._load_model()
        for item in self.include_items:
            if item.get('is_fixed', False): continue
            name = item['name']
            pos = item.get('pos', (0, 0, 0))
            quat = item.get("quat", (1, 0, 0, 0))

            self.set_item_pos(name, pos)
            self.set_item_rot(name, quat)

        self._init_control_info()

        self._max_iter = max_iter
        self._cur_iter = 0

        if view:
            self._launch_viewer(_CAM_WIDTH, _CAM_HEIGHT)
        else:
            self._viewer = None


    @classmethod
    def load_config(cls, config):
        mode = config.get("mode", "joint_angle")
        obs_include = config.get("obs_include", [])
        items = config.get("items", [])
        include_files = config.get("include_files", [])
        include_items = config.get("include_items", [])
        im_dims = config.get("image_dimensions", (_CAM_WIDTH. _CAM_HEIGHT))
        sim_freq = config.get("sim_freq", 25)
        ts = config.get("mjc_timestep", 0.002)
        view = config.get("view", False)
        max_iter = config.get("max_iterations", 250)
        return cls(mode, obs_include, items, include_files, include_items, im_dims, sim_freq, ts, max_iter, view)


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


    def get_state(self):
        return self.physics.data.qpos.copy()


    def set_state(self, state):
        self.physics.data.qpos[:] = state
        self.physics.forward()


    def __getstate__(self):
        return self.physics.data.qpos.tolist()


    def __setstate__(self, state):
        self.physics.data.qpos[:] = state
        self.physics.forward()


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


    def get_pos_from_label(self, label, mujoco_frame=True):
        try:
            pos = self.get_item_pos(label, mujoco_frame)
        except:
            pos = None

        return pos


    def get_item_pos(self, name, mujoco_frame=True):
        model = self.physics.model
        try:
            ind = model.name2id(name, 'joint')
            adr = model.jnt_qposadr[ind]
            pos = self.physics.data.qpos[adr:adr+3].copy()
        except Exception as e:
            try:
                item_ind = model.name2id(name, 'body')
                pos = self.physics.data.xpos[item_ind].copy()
            except:
                item_ind = -1

        return pos


    def get_item_rot(self, name, mujoco_frame=True, to_euler=False):
        model = self.physics.model
        try:
            ind = model.name2id(name, 'joint')
            adr = model.jnt_qposadr[ind]
            rot = self.physics.data.qpos[adr+3:adr+7].copy()
        except Exception as e:
            try:
                item_ind = model.name2id(name, 'body')
                rot = self.physics.data.xquat[item_ind].copy()
            except:
                item_ind = -1

        if to_euler:
            rot = T.quaternion_to_euler(rot[0], rot[1], rot[2], rot[3])

        return rot


    def set_item_pos(self, name, pos, mujoco_frame=True, forward=True):
        item_type = 'joint'
        try:
            ind = self.physics.model.name2id(name, 'joint')
            adr = self.physics.model.jnt_qposadr[ind]
            old_pos = self.physics.data.qpos[adr:adr+3]
            self.physics.data.qpos[adr:adr+3] = pos
        except Exception as e:
            try:
                ind = self.physics.model.name2id(name, 'body')
                old_pos = self.physics.data.xpos[ind]
                self.physics.data.xpos[ind] = pos
                # self.physics.model.body_pos[ind] = pos
                # old_pos = self.physics.model.body_pos[ind]
                item_type = 'body'
            except:
                item_type = 'unknown'
                print('Could not shift item', name)

        if forward:
            self.physics.forward()


    def set_item_rot(self, name, rot, use_euler=False, mujoco_frame=True, forward=True):
        if use_euler:
            rot = T.euler_to_quaternion(rot[0], rot[1], rot[2])
        item_type = 'joint'
        try:
            ind = self.physics.model.name2id(name, 'joint')
            adr = self.physics.model.jnt_qposadr[ind]
            old_quat = self.physics.data.qpos[adr+3:adr+7]
            self.physics.data.qpos[adr+3:adr+7] = rot
        except Exception as e:
            try:
                ind = self.physics.model.name2id(name, 'body')
                old_quat = self.physics.data.xquat[ind]
                self.physics.data.xquat[ind] = rot
                # self.physics.model.body_pos[ind] = pos
                # old_pos = self.physics.model.body_pos[ind]
                item_type = 'body'
            except:
                item_type = 'unknown'
                print('Could not rotate item', name)

        if forward:
            self.physics.forward()


    def get_geom_dimensions(self, geom_type=enums.mjtGeom.mjGEOM_BOX, geom_ind=-1):
        '''
        Geom type options:
        mjGEOM_PLANE=0, mjGEOM_HFIELD=1, mjGEOM_SPHERE=2, mjGEOM_CAPSULE=3, mjGEOM_ELLIPSOID=4, mjGEOM_CYLINDER=5, mjGEOM_BOX=6, mjGEOM_MESH=7
        '''
        if geom_ind >= 0:
            return self.physics.model.geom_size[ind]

        inds = np.where(self.physics.model.geom_type == geom_type)
        return self.physics.model.geom_size[inds]


    def get_geom_positions(self, geom_type=enums.mjtGeom.mjGEOM_BOX, geom_ind=-1):
        '''
        Geom type options:
        mjGEOM_PLANE=0, mjGEOM_HFIELD=1, mjGEOM_SPHERE=2, mjGEOM_CAPSULE=3, mjGEOM_ELLIPSOID=4, mjGEOM_CYLINDER=5, mjGEOM_BOX=6, mjGEOM_MESH=7
        '''
        if geom_ind >= 0:
            return self.physics.model.geom_pos[ind]
            
        inds = np.where(self.physics.model.geom_type == geom_type)
        return self.physics.model.geom_pos[inds]


    def get_geom_rotations(self, geom_type=enums.mjtGeom.mjGEOM_BOX, geom_ind=-1):
        '''
        Geom type options:
        mjGEOM_PLANE=0, mjGEOM_HFIELD=1, mjGEOM_SPHERE=2, mjGEOM_CAPSULE=3, mjGEOM_ELLIPSOID=4, mjGEOM_CYLINDER=5, mjGEOM_BOX=6, mjGEOM_MESH=7
        '''
        if geom_ind >= 0:
            return self.physics.model.geom_quat[ind]
            
        inds = np.where(self.physics.model.geom_type == geom_type)
        return self.physics.model.geom_quat[inds]


    # def get_items_in_region()


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
        return self.get_obs()


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