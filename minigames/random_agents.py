# Copyright 2018 Benjamin Bueno (bbueno5000) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A collection of random agents
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from baselines import logger
from pysc2.agents import random_agent


class RandomAgent(random_agent.RandomAgent):
    """
    Generic random agent
    """

    def __init__(self):
        super(RandomAgent, self).__init__()
        self.results = {}
        self.results['agent_id'] = "RandomAgent"
        self.results['episode_data'] = {'episode_lengths': [], 'episode_rewards': []}

    def reset(self):
        super(RandomAgent, self).reset()
        self.results['episode_data']['episode_lengths'].append(self.steps)
        self.results['episode_data']['episode_rewards'].append(self.reward)
        self.reward = 0
        self.steps = 0
        logger.record_tabular("episodes", self.episodes)
        logger.dump_tabular()
