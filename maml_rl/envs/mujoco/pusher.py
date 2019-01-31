from mujoco_env import MujocoEnv
import numpy as np
import pickle
from gym import utils

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


class PusherEnv(MujocoEnv, utils.EzPickle):
    """
    Pusher Environment adapted from the MAESN repository.

    Has support for both a sparse reward and a dense reward.

    """

    FILE = 'pusher_env.xml'

    def __init__(self, choice=None, sparse = False , train = True):
        self.choice = choice
        if train:
            assert sparse is False
            self.all_goals = pickle.load(open("/Users/navneetmadhukumar/Downloads/pytorch-maml-rl/maml_rl/envs/goals/pusher_trainSet.pkl", "rb"))
        else:
            assert sparse is True
            self.all_goals = pickle.load(open('/Users/navneetmadhukumar/Downloads/pytorch-maml-rl/maml_rl/envs/goals/pusher_valSet.pkl', 'rb'))
        self.sparse = sparse
        super(PusherEnv, self).__init__()
        utils.EzPickle.__init__(self)

    def sample_goals(self, num_goals):
        return np.array([np.random.randint(0, num_goals*5) for i in range(num_goals)])

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
        curr_qpos = blockpositions

        curr_qvel = self.sim.data.qvel.copy()
        curr_qvel = np.zeros_like(curr_qvel)
        self.set_state(curr_qpos, curr_qvel)
        self.sim.forward()
        obs = self.get_current_obs()
        # for i in range(10):
        #     self.model.step()
        return obs

    def get_current_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:3],
            self.sim.data.geom_xpos[-6:-1, :2].flat,
            self.sim.data.qvel.flat,
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        blockchoice = self.goal[0]
        curr_block_xidx = int(3 + 2*blockchoice)
        curr_block_yidx = int(4 + 2*blockchoice)
        curr_gripper_pos = self.sim.data.site_xpos[0, :2]
        curr_block_pos = np.array([next_obs[curr_block_xidx], next_obs[curr_block_yidx]])
        goal_pos = self.goal[1:3]
        dist_to_block = np.linalg.norm(curr_gripper_pos - curr_block_pos)
        block_dist = np.linalg.norm(goal_pos - curr_block_pos)
        goal_dist = np.linalg.norm(goal_pos)
        
        if self.sparse and block_dist > 0.2:
            reward = - 5*goal_dist
        else:
            reward = -5*block_dist 
        done = False
        info = {'distance_to_block': dist_to_block}
        return next_obs, reward, done, info


if __name__ == '__main__':
    # Testing the new environment
    env = PusherEnv()
    env.reset()
    env.render()
    num_simulation_steps = 100
    for _ in range(num_simulation_steps):
        next_obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()
