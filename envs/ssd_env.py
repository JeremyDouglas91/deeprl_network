import matplotlib
import configparser
import logging
import gym, ray
import numpy as np
import pandas as pd
import os

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv

matplotlib.use('agg')
gym.logger.set_level(40) # supress warnings
DEFULT_MAX_STEPS = 1000

class SSDEnv(gym.Env, ray.rllib.env.MultiAgentEnv):
    def __init__(self, config):
        self.scenario = config.get("scenario")
        self.name = config.get("scenario") # the trainer needs this, annoyingly. 
        self.agent = config.get("agent")
        self.n_agent = config.getint("n_agent")
        self.agent_tags = ['agent-%s'%(i) for i in range(self.n_agent)]

        if self.scenario.lower().endswith("harvest"):
            self.env = HarvestEnv(num_agents=self.n_agent)
        elif self.scenario.lower().endswith("cleanup"):
            self.env = CleanupEnv(num_agents=self.n_agent)
        else:
            raise ValueError(f"{config['env']} is an invalid value for config[\"env\"].")

        self.coop_gamma = config.getfloat('coop_gamma') # Spatial discount factor
        self.dt = config.getfloat('control_interval_sec') # control step (always 1 for cleanup and harvest)
        self.T = int(config.getint('episode_length_sec') / self.dt) # max steps
        self.steps = 0 # episode step counter

        # Initialise graph structure etc.
        self._init_space()

        # Set seeds for reproducible experiments
        self.init_test_seeds(config)
        np.random.seed(self.seed)

        # create a csv file to store data
        metrics = ['action','reward']
        col_names = ['ep_step']+[agent+"_"+metric for agent in self.agent_tags for metric in metrics]
        self.df_train_log = pd.DataFrame(data={}, columns=col_names)
        self.train_log_path = os.path.join(os.getcwd(), 'output/data/train_data_log.csv')
        self.init_data_log()
        
    def _init_space(self):
        # action and observatrion space
        env_window_size = 2 * self.env.agents[list(self.env.agents.keys())[0]].view_len + 1
        obs_shape =  (env_window_size, env_window_size, 3)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0,
            shape=obs_shape,
            dtype=np.float32
        )

        # actions (number per agent, list of number of actions for each agent, possible actions/map)
        self.n_a = self.action_space.n
        self.n_a_ls = [self.n_a]*self.n_agent
        self.a_map = [i for i in range(self.n_a)] # probably unnecessary since actions are inetegers...
        logging.info('action to h_go map:\n %r' % self.a_map)

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

    def init_test_seeds(self, config):
        self.seed = config.getint('seed')
        self.test_seeds = [int(s) for s in config.get('test_seeds').split(',')]
        self.test_num = len(self.test_seeds )

    def init_data_log(self):
        self.df_train_log.to_csv(path_or_buf=self.train_log_path, header=True)

    def update_data_log(self):
        self.df_train_log.to_csv(path_or_buf=self.train_log_path, mode='a', header=False)
        self.df_train_log.drop(self.df_train_log.index, inplace=True)

    def reset(self):
        self.steps = 0 # episode step counter
        self.rewards = [0] # to keep track of global rewards
        obs = self.env.reset() # dictionary
        obs = list(obs.values())

        return obs

    def step(self, action):
        """
        return:
        obs: list of numpy arrays
        reward: float if coop_gamma < 0 else numpy array ?
        done: boolean
        global_reward: float
        """
        # increment episode step counter
        self.steps += 1
        
        # step in environment
        action_dict = dict(zip(self.agent_tags, action)) # format agents actions as dictionary for env
        obs, reward, done, info = self.env.step(action_dict)

        obs = list(obs.values())
        done = self.steps >= self.T 
        reward = list(reward.values())
        global_reward = np.sum(reward)
        self.rewards.append(global_reward)

        # log data
        data = [self.steps]
        for a,r in zip(action,reward):
            data.append(a)
            data.append(r)
        self.df_train_log = self.df_train_log.append(dict(zip(self.df_train_log.columns, data)), 
                                                    ignore_index=True)
        if done:
            self.update_data_log()

        # no spatial discounting 
        if(self.coop_gamma<0):
            reward = global_reward
        else:
            reward = np.array(reward) # discounting happens in buffer class later (def _add_st_R_Adv(self, R, dt):)

        return obs, reward, done, global_reward

    def get_rewards(self, agent_id):
        return self.rewards[agent_id]

    def get_next_obs(self, agent_id):
        return self.observations[agent_id]

    def get_observations(self):
        return self.observations

    def get_fingerprint(self):
        return self.fp

    def update_fingerprint(self, fp):
        self.fp = fp

    def get_neighbor_action(self, action):
        naction = []
        for i in range(self.n_agent):
            naction.append(action[self.neighbor_mask[i] == 1])
        return naction

    def terminate(self):
        return

