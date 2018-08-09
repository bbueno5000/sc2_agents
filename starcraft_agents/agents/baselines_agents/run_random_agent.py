import absl.app as app
import argparse
import gym
import gym_sc2
import random_agent
import sys

def main(argv):
    env = gym.make('MoveToBeacon-bbueno5000-v0')
    outdir = '/tmp/random-agent-results'
    env = gym.wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = random_agent.RandomAgent(env.action_space)
    episode_count = 100
    reward = 0
    done = False
    for _ in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
    env.close()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    # args = parser.parse_args()
    gym.logger.set_level(gym.logger.INFO)
    app.run(main)
