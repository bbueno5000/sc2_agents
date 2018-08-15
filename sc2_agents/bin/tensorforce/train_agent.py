# MIT License
#
# Copyright (c) 2018 Benjamin Bueno (bbueno5000)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
OpenAI gym execution of TensorForce.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
from json import load
from logging import basicConfig as logging_basicConfig
from logging import getLogger
from logging import INFO
from os import mkdir
from os import path
from gym_sc2 import envs
from tensorforce import TensorForceError
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
from time import time

FLAGS = flags.FLAGS
flags.DEFINE_string('agent_config', None, "Agent configuration file")
flags.DEFINE_bool('debug', False, "Show debug outputs")
flags.DEFINE_bool('deterministic', False, "Choose actions deterministically")
flags.DEFINE_integer('num_episodes', 10, "Number of episodes")
flags.DEFINE_string('gym_id', None, "Id of the Gym environment")
flags.DEFINE_string('job', None, "For distributed mode: The job type of this agent.")
flags.DEFINE_string('load', None, "Load agent from this dir")
flags.DEFINE_integer('max_episode_timesteps', None, "Maximum number of timesteps per episode")
flags.DEFINE_string('monitor', None, "Save results to this directory")
flags.DEFINE_bool('monitor_safe', False, "Do not overwrite previous results")
flags.DEFINE_integer('monitor_video', 0, "Save video every x steps (0 = disabled)")
flags.DEFINE_string('network', None, "Network specification file")
flags.DEFINE_string('save', None, "Save agent to this dir")
flags.DEFINE_integer('save_episodes', 100, "Save agent every x episodes")
flags.DEFINE_float('sleep', None, "Slow down simulation by sleeping for x seconds (fractions allowed).")
flags.DEFINE_integer('task', 0, "For distributed mode: The task index of this agent.")
flags.DEFINE_bool('test', False, "Test agent without learning.")
flags.DEFINE_integer('timesteps', None, "Number of timesteps")
flags.DEFINE_bool('visualize', False, "Enable OpenAI Gym's visualization")

def main(argv):
    logging_basicConfig(level=INFO)
    logger = getLogger(__file__)
    logger.setLevel(INFO)

    environment = OpenAIGym(
        gym_id='MoveToBeacon-bbueno5000-v0',
        monitor=FLAGS.monitor,
        monitor_safe=FLAGS.monitor_safe,
        monitor_video=FLAGS.monitor_video,
        visualize=FLAGS.visualize)

    # if FLAGS.agent_config is not None:
    #     with open(FLAGS.agent_config, 'r') as fp:
    #         agent_config = json.load(fp=fp)
    # else:
    #     raise TensorForceError(
    #         "No agent configuration provided.")

    # if FLAGS.network is not None:
    #     with open(FLAGS.network, 'r') as fp:
    #         network = json.load(fp=fp)
    # else:
    #     network = None
    #     logger.info(
    #         "No network configuration provided.")

    network_spec = [
        dict(type='flatten'),
        dict(type='dense', size=32),
        dict(type='dense', size=32)
        ]

    agent = PPOAgent(
        states=environment.states,
        actions=environment.actions,
        network=network_spec
        )

    if FLAGS.load:
        load_dir = path.dirname(FLAGS.load)
        if not path.isdir(load_dir):
            raise OSError(
                "Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(FLAGS.load)

    if FLAGS.save:
        save_dir = path.dirname(FLAGS.save)
        if not path.isdir(save_dir):
            try:
                mkdir(save_dir, 0o755)
            except OSError:
                raise OSError(
                    "Cannot save agent to dir {} ()".format(save_dir))

    if FLAGS.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1)

    if FLAGS.debug:
        report_episodes = 1
    else:
        report_episodes = 100

    logger.info(
        "Starting {agent} for Environment {env}".format(
            agent=agent, env=environment))

    def episode_finished(r, id_):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time() - r.start_time)
            logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                r.agent.episode, r.episode_timestep, steps_per_second))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {:0.2f}".format(
                sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {:0.2f}".format(
                sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
        if FLAGS.save and FLAGS.save_episodes is not None and not r.episode % FLAGS.save_episodes:
            logger.info("Saving agent to {}".format(FLAGS.save))
            r.agent.save_model(FLAGS.save)
        return True

    runner.run(
        num_timesteps=FLAGS.timesteps,
        num_episodes=FLAGS.num_episodes,
        max_episode_timesteps=FLAGS.max_episode_timesteps,
        deterministic=FLAGS.deterministic,
        episode_finished=episode_finished,
        testing=FLAGS.test,
        sleep=FLAGS.sleep)

    runner.close()

    logger.info("Learning completed.")
    logger.info("Total episodes: {ep}".format(ep=runner.agent.episode))

if __name__ == '__main__':
    app.run(main)

# python openai_gym/move_to_beacon.py --gym_id SC2MoveToBeacon-bbueno5000-v0 --agent_config openai_gym/configs/vpg.json --network openai_gym/configs/mlp2_network.json
