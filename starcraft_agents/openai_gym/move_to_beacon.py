# Copyright 2018 Benjamin Bueno (bbueno5000). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OpenAI gym execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import json
import logging
import os
import gym_sc2
from tensorforce import TensorForceError
from tensorforce.agents import RandomAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import time

FLAGS = flags.FLAGS
flags.DEFINE_string('agent_config', None, "Agent configuration file")
flags.DEFINE_bool('debug', False, "Show debug outputs")
flags.DEFINE_bool('deterministic', False, "Choose actions deterministically")
flags.DEFINE_integer('episodes', None, "Number of episodes")
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
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

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

    agent = RandomAgent(environment.states, environment.actions)

    if FLAGS.load:
        load_dir = os.path.dirname(FLAGS.load)
        if not os.path.isdir(load_dir):
            raise OSError(
                "Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(FLAGS.load)

    if FLAGS.save:
        save_dir = os.path.dirname(FLAGS.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
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
            steps_per_second = r.timestep / (time.time() - r.start_time)
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
        num_episodes=FLAGS.episodes,
        max_episode_timesteps=FLAGS.max_episode_timesteps,
        deterministic=FLAGS.deterministic,
        episode_finished=episode_finished,
        testing=FLAGS.test,
        sleep=FLAGS.sleep)

    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))

if __name__ == '__main__':
    app.run(main)

# python openai_gym/move_to_beacon.py --gym_id SC2MoveToBeacon-bbueno5000-v0 --agent_config openai_gym/configs/vpg.json --network openai_gym/configs/mlp2_network.json
