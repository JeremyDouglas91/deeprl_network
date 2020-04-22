"""
A wrapper class for the sequential social dilema (SSD) 
environments Cleanup and Harvest.
@author: Arnu Pretorius, Elan, Jeremy du Plessis
"""
import configparser
import gym
import logging
import matplotlib
import numpy as np
import pandas as pd
import ray

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv

matplotlib.use('agg') # what are we using this for?
gym.logger.set_level(40) # supress precision warning when using 32-bit float
DEFULT_MAX_STEPS = 1000

class SSDEnv(gym.Env, ray.rllib.env.MultiAgentEnv):
    def __init__(self, config):
        """
        Constructor for ssd env wrapper:
        -------------------------------
        - initialise instance variables from config file parameters.
        - load core environment.
        - call additional intialisation functions.
        """
        self.scenario = config.get("scenario") # e.g. ssd_cleanup
        self.name = config.get("scenario") # annoyingly, the nmarl code needs this. 
        self.agent = config.get("agent") # e.g. ia2c_fp
        self.n_agent = config.getint("n_agent") # number of agents 
        self.agent_tags = ['agent-%s'%(i) for i in range(self.n_agent)]
        self.coop_gamma = config.getfloat('coop_gamma') # Spatial discount factor
        self.dt = config.getfloat('control_interval_sec') # control interval (always 1 for cleanup and harvest)
        self.T = int(config.getint('episode_length_sec') / self.dt) # max steps
        assert (self.T == DEFULT_MAX_STEPS)
        self.cur_episode = 0 # episode counter 
        self.is_record = False # record data for each step (only True during evaluation)
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
        self.init_test_seeds(test_seeds)
    
    def init_data(self, is_record, record_stats, output_path):
        """
        params
        ------
        is_record:    boolean, whether or not to record metrics during evaluation.
        record_stats: boolean, not used for ssd.
        output_path:  string, location to save .csv file with metrics.
        """
        self.is_record = is_record
        self.output_path = output_path
        if self.is_record:
            self.control_data = []

    def _init_space(self):
        """
        Initialise objects in the task space:
        -------------------------------------
        - env action & observation space
        - num actions per agent
        - agent adjacency matrix (graph)
        - agent distance matrix (graph)
        - observation dimension per agent (15,15,3)
        - agent finger prints (distribution of actions over state)
        """
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
        self.n_s_ls = []
        for i in range(self.n_agent):
            if self.agent.startswith('ma2c'):
                num_n = 1
            else:
                num_n = 1 + np.sum(self.neighbor_mask[i])
            self.n_s_ls.append((num_n, obs_shape[0], obs_shape[1], obs_shape[2]))

        # initialise finger print (policy distribution over actions per agent)
        self.fp = np.ones((self.n_agent, self.n_a)) / self.n_a

    def init_test_seeds(self, test_seeds):
        """
        params:
        -------
        test_seeds: list (int), seeds used in model evaluation.
        """
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def collect_tripinfo(self):
        # not applicable to ssd.
        return

    def _log_control_data(self, action, global_reward):
        """
        Log control data at each step during evaluation.

        params:
        -------
        action: list (int), agent actions.
        global_reward: float, collective reward (undiscounted).
        """
        action_r = ','.join(['%d' % a for a in action])
        cur_control = {'episode': self.cur_episode,
                       'step': self.t,
                       'action': action_r,
                       'reward': global_reward}
        self.control_data.append(cur_control)

    def get_fingerprint(self):
        """
        Returns agents fingerprints (policies).
        
        returns:
        --------
        self.fp: 2D list (float), one policy (distribution 
                 over actions) per agent.
        """
        return self.fp

    def get_neighbor_action(self, action):
        """
        Get actions of each agents neighbour in 
        the graph.

        params:
        -------
        action: list (int), agent actions for a single
                time step.

        returns:
        --------
        naction: 2D list (int), neighbour actions for 
                 each agent.  
        """
        naction = []
        for i in range(self.n_agent):
            naction.append(action[self.neighbor_mask[i] == 1])
        return naction

    def _get_state(self, obs_env):
        """
        Return the apprpriate observations to the agents
        depending on the type of algoithm being run.

        params
        ------
        obs_env: 4D list (float), list of observations 
                 from the core environment, one per agent.

        returns:
        --------
        state: ND list (float), list of appropriate 
               observations, one per agent.
        """
        state = []
        obs_env = obs_env/self.observation_space.high
        for i in range(self.n_agent):
            local_obs = obs_env[i]
            if self.agent.startswith('ia2c'):
                imgs = [local_obs]

                if not self.agent == 'ia2c_fp': # ia2c
                    for j in np.where(self.neighbor_mask[i] == 1)[0]:
                        imgs.append(obs_env[j])
                    imgs = np.array(imgs, dtype=np.float32)
                    fps = np.array([], dtype=np.float32)

                else: # ia2c_fp
                    fps = []
                    for j in np.where(self.neighbor_mask[i] == 1)[0]:
                        imgs.append(obs_env[j])
                        fps.append(self.fp[j])
                    imgs = np.array(imgs, dtype=np.float32)
                    fps = np.concatenate(fps).astype(np.float32) 

                agent_obs = [imgs, fps]

            else: # ma2c
                agent_obs = local_obs.astype(np.float32)

            state.append(agent_obs)
        return state

    def output_data(self):
        """
        Save control data from evaluation to disk.
        """
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))

    def reset(self, gui=False, test_ind=-1):
        """
        Reset environment state, set new random seeds, 
        reset metrics, update episode counter etc.
        
        params:
        -------
        gui: not used for ssd environments.

        returns:
        --------
        obs: ND list (float), appropriate observations, 
             one per agent. 
        """
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
        params:
        -------
        action: list (int), one action for each of self.n_agent

        returns:
        --------
        obs: ND list (float), appropriate observations, 
             one per agent. 
        reward: global_reward if coop_gamma < 0 else numpy array
                of individual rewards.
        done: boolean.
        global_reward: float, collective reward (undiscounted)
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
        """
        not used in ssd.
        """
        return

    def update_fingerprint(self, fp):
        """
        Sets agents fingerprints (policies); distributions
        over actions given the current state.
        """
        self.fp = fp

