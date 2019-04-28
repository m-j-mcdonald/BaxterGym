import sys
import traceback

from baxter_gym.envs.mjc_env import *
from baxter_gym.envs.baxter_cloth_env import *
from baxter_gym.envs.baxter_mjc_env import *
from baxter_gym.envs.baxter_rope_env import *
from baxter_gym.envs.hsr_mjc_env import *
from baxter_gym.envs.hsr_ros_env import *
try:
    from baxter_gym.envs.hsr_ros_env import *
except Exception as e:
    traceback.print_exception(*sys.exc_info())
