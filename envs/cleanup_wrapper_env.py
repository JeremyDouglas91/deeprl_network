import matplotlib
matplotlib.use('agg')
​
import gym, ray
import numpy as np
​
from social_dilemmas.envs.cleanup import CleanupEnv
​
DEFULT_MAX_STEPS = 1000
​
class SSD(gym.Env, ray.rllib.env.MultiAgentEnv):
    def __init__(self, config):
        if config["env"].lower() == "harvest":
            self.env = HarvestEnv(num_agents=config["num_agents"])
        elif config["env"].lower() == "cleanup":
            self.env = CleanupEnv(num_agents=config["num_agents"])
        else:
            raise ValueError(f"{config['env']} is an invalid value for config[\"env\"].")
​
        env_window_size = 2 * self.env.agents[list(self.env.agents.keys())[0]].view_len + 1
​
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0,
            shape=(
                3,
                env_window_size,
                env_window_size
            ),
            dtype=np.float32
        )
​
        self.max_steps = config["max_steps"] if "max_steps" in config else DEFULT_MAX_STEPS
        self.steps = 0
        self.observations = None
        self.rewards = None
        self.dones = None
​
    def reset(self):
        self.steps = 0
        self.observations = None
        self.rewards = None
        self.dones = None
​
        obs = self.env.reset()
        for agent_id, agent_obs in obs.items():
            obs[agent_id] = agent_obs.transpose(2, 0, 1)
        self.observations = obs
        return obs
​
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for agent_id, agent_obs in obs.items():
            obs[agent_id] = agent_obs.transpose(2, 0, 1)
​
        _done = self.steps >= self.max_steps
        for key in done:
            done[key] = _done
        self.observations = obs
        self.rewards = reward
        self.dones = done
        return obs, reward, done, info
​
    def get_rewards(self, agent_id):
        return self.rewards[agent_id]
​
    def get_next_obs(self, agent_id):
        return self.observations[agent_id]
​
    def get_observations(self):
        return self.observations
​