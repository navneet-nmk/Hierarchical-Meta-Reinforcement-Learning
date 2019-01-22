import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides
import pickle

def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param

def generate_goals(num_goals):
    radius = 2.0 
    angle = np.random.uniform(0, np.pi, size=(num_goals,))
    xpos = radius*np.cos(angle)
    ypos = radius*np.sin(angle)
    return np.concatenate([xpos[:, None], ypos[:, None]], axis=1)

#num_goals = 40
#goals = generate_goals(num_goals)
#import pickle
#pickle.dump(goals, open("goals_wheeled.pkl", "wb"))

class WheeledEnv(MujocoEnv, Serializable):

    FILE = 'wheeled.xml'

    def __init__(self, goal=None, sparse = False , train = True,  *args, **kwargs):
        self._goal_idx = goal 
        if train :
            assert sparse == False
            self.goals =  pickle.load(open("/root/code/rllab/rllab/envs/goals/wheeled_trainSet.pkl", "rb"))
        else:
            assert sparse == True
            self.goals = pickle.load(open('/root/code/rllab/rllab/envs/goals/wheeled_valSet.pkl' , 'rb'))
        
        self.sparse = sparse
        super(WheeledEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])

    def sample_goals(self, num_goals):
        return np.random.choice(np.array(range(num_goals*5)), num_goals)
        #return np.array(range(num_goals))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        goal_idx = reset_args
        if goal_idx is not None:
            self._goal_idx = goal_idx
        elif self._goal_idx is None:
            self._goal_idx = np.random.randint(1)
        self.reset_mujoco(init_state)
        body_pos = self.model.body_pos.copy()
        body_pos[-1][:2] = self.goals[self._goal_idx]
        self.model.body_pos = body_pos
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs

    def step(self, action):
        
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
       
        if self.sparse and np.linalg.norm(next_obs[:2] -self.goals[self._goal_idx] ) > 0.8 :
            reward = -np.linalg.norm(self.goals[self._goal_idx]) - ctrl_cost
        else:
            reward = -np.linalg.norm(next_obs[:2] - self.goals[self._goal_idx]) - ctrl_cost 
           
        done = False
        return Step(next_obs, reward, done)
