from absl import app
from argparse import ArgumentParser
from gym import make
from gym import wrappers
from gym_sc2 import envs
from sc2_agents.agents import random_agents
from run_agent import run_agent


def main(argv):
    env = make('MoveToBeacon-bbueno5000-v0')
    agent = random_agents.RandomAgent002(env.action_space)
    outdir = 'random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    episode_count = 10
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
    # parser = ArgumentParser(description=None)
    # parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    # args = parser.parse_args()
    app.run(main)
