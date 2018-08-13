from absl import app
from argparse import ArgumentParser
from gym import make
from gym_sc2 import envs
from sc2_agents.agents import random_agents
from run_agent import run_agent


def main(argv):
    env = make('MoveToBeacon-bbueno5000-v0')
    agent = random_agents.RandomAgent002(env.action_space)
    run_agent(agent, env)


if __name__ == '__main__':
    # parser = ArgumentParser(description=None)
    # parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    # args = parser.parse_args()
    app.run(main)
