import gym
import gym_sc2
import numpy as np

_NO_OP = 0
_PLAYER_NEUTRAL = 3

class MoveToBeacon1d:

    def __init__(self, visualize=False, step_mul=None) -> None:
        self.env_name = "MoveToBeacon-bbueno5000-v0"
        self.visualize = visualize
        self.step_mul = step_mul

    def get_action(self, env, obs):
        neutral_y, neutral_x = (obs[0] == _PLAYER_NEUTRAL).nonzero()
        if not neutral_y.any():
            raise Exception('Beacon not found!')
        target = [int(neutral_x.mean()), int(neutral_y.mean())]
        return np.ravel_multi_index(target, obs.shape[1:])

    def run(self, num_episodes=1):
        env = gym.make(self.env_name)
        env.settings['visualize'] = self.visualize
        env.settings['step_mul'] = self.step_mul
        episode_rewards = np.zeros((num_episodes, ), dtype=np.int32)
        for ix in range(num_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = self.get_action(env, obs)
                obs, _, done, _ = env.step(action)
            episode_rewards[ix] = env.episode_reward
        env.close()
        return episode_rewards
