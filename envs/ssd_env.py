import matplotlib
import configparser
import logging
import gym, ray
import numpy as np
import pandas as pd

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv

matplotlib.use('agg')
DEFULT_MAX_STEPS = 1000

class SSDEnv(gym.Env, ray.rllib.env.MultiAgentEnv):
    def __init__(self, config):
        self.scenario = config.get("scenario")
        self.name = config.get("scenario") # the trainer needs this, annoyingly. 
        self.agent = config.get("agent")
        self.n_agent = config.getint("n_agent")
        self.agent_tags = ['agent-%s'%(i) for i in range(self.n_agent)]
        self.coop_gamma = config.getfloat('coop_gamma') # Spatial discount factor
        self.dt = config.getfloat('control_interval_sec') # control step (always 1 for cleanup and harvest)
        self.T = int(config.getint('episode_length_sec') / self.dt) # max steps
        self.cur_episode = 0 # episode counter 
        self.is_record = False # record data for each step (usually only during evaluation)
        self.train_mode = True

        if self.scenario.lower().endswith("harvest"):
            self.env = HarvestEnv(num_agents=self.n_agent)
        elif self.scenario.lower().endswith("cleanup"):
            self.env = CleanupEnv(num_agents=self.n_agent)
        else:
            raise ValueError(f"{config['env']} is an invalid value for config[\"env\"].")

        # Initialise graph structure etc.
        self._init_space()

        # Set seeds for reproducible experiments
        self.seed = config.getint('seed')
        test_seeds = [int(s) for s in config.get('test_seeds').split(',')]
        self.init_test_seeds(test_seeds) # this function call is annoying, but the evaluation function uses it.
    
    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.output_path = output_path
        if self.is_record:
            self.control_data = []

    def _init_space(self):
        # action and observatrion space
        env_window_size = 2 * self.env.agents[list(self.env.agents.keys())[0]].view_len + 1
        obs_shape =  (env_window_size, env_window_size, 3) # TF needs shape (height, width, channels)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0,
            shape=obs_shape,
            dtype=np.float32
        )

        # actions (number per agent, list of number of actions for each agent, possible actions/map)
        self.n_a = self.action_space.n
        self.n_a_ls = [self.n_a]*self.n_agent

        # neighbor & distance graphs
        # start with a fully connected graph, K_{n_agent}
        self.neighbor_mask = np.ones((self.n_agent, self.n_agent), int)
        self.distance_mask = np.ones((self.n_agent, self.n_agent), int)
        np.fill_diagonal(self.neighbor_mask,0)
        np.fill_diagonal(self.distance_mask,0)

        # dimensions of local observation, 1 per agent 
        self.n_s_ls = [obs_shape for i in range(self.n_agent)]

        # initialise finger print (policy distribution over actions per agent)
        self.fp = np.ones((self.n_agent, self.n_a)) / self.n_a

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def collect_tripinfo(self):
        # not applicable to ssd.
        return

    def _log_control_data(self, action, global_reward):
        action_r = ','.join(['%d' % a for a in action])
        cur_control = {'episode': self.cur_episode,
                       'step': self.t,
                       'action': action_r,
                       'reward': global_reward}
        self.control_data.append(cur_control)

    def get_fingerprint(self):
        return self.fp

    def get_neighbor_action(self, action):
        naction = []
        for i in range(self.n_agent):
            naction.append(action[self.neighbor_mask[i] == 1])
        return naction

    def _get_state(self, obs_env):
        state = []
        for i in range(self.n_agent):
            img_obs_norm = obs_env[i]/self.observation_space.high
            if self.agent == 'ia2c_fp':
                n_fps = []
                # finger prints must be attached at the end of the state array
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    n_fps.append(self.fp[j])
                n_fps = np.concatenate(n_fps) # all finger prints from neighbours
                cur_state = [img_obs_norm.astype(np.float32), n_fps.astype(np.float32)]
            elif self.agent == 'ia2c':
                cur_state = [img_obs_norm.astype(np.float32), np.array([]).astype(np.float32)]
            else:
                cur_state = img_obs_norm.astype(np.float32)
            state.append(cur_state)
        return state

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))

    def reset(self, gui=False, test_ind=-1):
        # set seed 
        if (self.train_mode):
            seed = self.seed
        elif (test_ind < 0):
            seed = self.seed-1
        else:
            seed = self.test_seeds[test_ind]
        np.random.seed(seed)
        self.seed += 1

        self.cur_episode += 1 
        self.t = 0 # step counter for each episode
        self.rewards = [0] # to keep track of global rewards
        obs = self.env.reset() # dictionary
        obs = list(obs.values())

        obs = self._get_state(obs) # new 

        return obs

    def step(self, action):
        """
        parameters
        ----------
        action: list (int), one action for each of self.n_agent

        return
        ------
        obs: list of numpy arrays
        reward: float if coop_gamma < 0 else numpy array 
        done: boolean
        global_reward: float
        """
        # increment episode step counter
        self.t += 1
        
        # format agents actions as dictionary (required for ssd env)
        action_dict = dict(zip(self.agent_tags, action)) 

        # step in environment
        obs, reward, done, info = self.env.step(action_dict)

        # items to return
        obs = list(obs.values())
        obs = self._get_state(obs) # new 
        done = self.t >= self.T 
        reward = list(reward.values())
        global_reward = np.sum(reward)

        self.rewards.append(global_reward)
        if(self.coop_gamma<0): # no spatial discounting 
            reward = global_reward
        else: # discounting happens in buffer class later (def _add_st_R_Adv(self, R, dt):)
            reward = np.array(reward) 

        # record data from step
        if self.is_record:
            self._log_control_data(action, global_reward)

        return obs, reward, done, global_reward

    def terminate(self):
        return

    def update_fingerprint(self, fp):
        self.fp = fp

