from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from .mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import pickle

#num_tasks = 20

def save_goal_samples(num_tasks, filename):
    all_goals = []
    blockarr = np.array(range(5))
    for _ in range(num_tasks):
        blockpositions = np.zeros((13, 1))
        blockpositions[0:3] = 0
        positions_sofar = []
        np.random.shuffle(blockarr)
        for i in range(5):
            xpos = np.random.uniform(.35, .65)
            ypos = np.random.uniform(-.5 + 0.2*i, -.3 + 0.2*i)
            blocknum = blockarr[i]
            blockpositions[3+2*blocknum] = -0.2*(blocknum + 1) + xpos
            blockpositions[4+2*blocknum] = ypos
            curr_pos = np.array([xpos, ypos])

        for i in range(5):
            blockchoice = blockarr[i]
            goal_xpos = np.random.uniform(.75, .95)
            goal_ypos = np.random.uniform(-.5 + .2*i, -.3 + .2*i)
            curr_goal = np.concatenate([[blockchoice], np.array([goal_xpos, goal_ypos]), blockpositions[:,0]])
            all_goals.append(curr_goal)
    all_goals = np.asarray(all_goals)
    np.random.shuffle(all_goals)
    pickle.dump(np.asarray(all_goals), open(filename, "wb"))
    return np.asarray(all_goals)

#save_goal_samples(20, "pusher_trainSet_20Tasks.pkl")

# all_goals = save_goal_samples(num_tasks)
# import IPython
# IPython.embed()
class PusherEnv(MujocoEnv, Serializable):

    FILE = 'pusher_env.xml'

    def __init__(self, choice=None, sparse = False , train = True):
        self.choice = choice
        if train :
            assert sparse == False
            self.all_goals =  pickle.load(open("/root/code/rllab/rllab/envs/goals/pusher_trainSet.pkl", "rb"))
        else:
            assert sparse == True
            self.all_goals = pickle.load(open('/root/code/rllab/rllab/envs/goals/pusher_valSet.pkl' , 'rb'))
        self.sparse = sparse
        #all_goals = pickle.load(open("/home/russellm/generativemodel_tasks/maml_rl_fullversion/rllab/envs/mujoco/pusher_trainSet_100Tasks.pkl", "rb"))
        super(PusherEnv, self).__init__()

    def sample_goals(self, num_goals):
        return np.array([np.random.randint(0, num_goals*5) for i in range(num_goals)])

       #return np.asarray(range(num_goals))
       # return np.random.choice(range(num_tasks*5), size=(num_goals,))

    def reset(self, reset_args=None):
        choice = reset_args
        if choice is not None:
            self.choice = choice
        elif self.choice is None:
            self.choice = np.random.randint(1)

        self.goal = self.all_goals[self.choice]

        blockchoice = self.goal[0]
        goal_pos = self.goal[1:3]
        blockpositions = self.goal[3:]

        body_pos = self.model.body_pos.copy()
        body_pos[-1][0] = goal_pos[0]
        body_pos[-1][1] = goal_pos[1]
        self.model.body_pos = body_pos
        curr_qpos = blockpositions
        self.model.data.qpos = curr_qpos

        curr_qvel = self.model.data.qvel.copy()
        curr_qvel = np.zeros_like(curr_qvel)
        self.model.data.qvel = curr_qvel
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        # for i in range(10):
        #     self.model.step()
        return obs

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:3],
            self.model.data.geom_xpos[-6:-1, :2].flat,
            self.model.data.qvel.flat,
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        blockchoice = self.goal[0]
        curr_block_xidx = int(3 + 2*blockchoice)
        curr_block_yidx = int(4 + 2*blockchoice)
        #TODO: Maybe need to change angle here
        curr_gripper_pos = self.model.data.site_xpos[0, :2]
        curr_block_pos =  np.array([next_obs[curr_block_xidx], next_obs[curr_block_yidx]])
        goal_pos = self.goal[1:3]
        dist_to_block = np.linalg.norm(curr_gripper_pos -  curr_block_pos)
        block_dist = np.linalg.norm(goal_pos - curr_block_pos)
        goal_dist = np.linalg.norm(goal_pos)
        
        if self.sparse and block_dist > 0.2:
            reward = - 5*goal_dist
        else:
            reward = -5*block_dist 
        done = False
        return Step(next_obs, reward, done)


# if __name__ == "__main__":
#     env = PusherEnvRandomized()
#     env.reset()
#     env.step(np.zeros(3))
#     env.render()
#     import IPython
#     IPython.embed()
