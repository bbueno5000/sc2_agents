# Copyright 2017 Google Inc. All Rights Reserved.
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

A collection of random agents to use as a baseline.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from baselines import logger
from numpy.random import choice as np_choice
from numpy.random import randint as np_randint
from pysc2.agents import base_agent
from pysc2.lib import actions


class RandomAgent001(base_agent.BaseAgent):

    def step(self, obs):
        super(RandomAgent001, self).step(obs)
        function_id = np_choice(obs.observation['available_actions'])
        args = [[np_randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)


# class RandomAgent002(RandomAgent001):
#     """
#     A random agent with graphing features.
#     """
#     def __init__(self):
#         super(RandomAgent002, self).__init__()
#         self.results = {}
#         self.results['agent_id'] = "RandomAgent002"
#         self.results['episode_data'] = {'episode_lengths': [], 'episode_rewards': []}

#     def reset(self):
#         super(RandomAgent002, self).reset()
#         self.results['episode_data']['episode_lengths'].append(self.steps)
#         self.results['episode_data']['episode_rewards'].append(self.reward)
#         self.reward = 0
#         self.steps = 0
#         logger.record_tabular("episodes", self.episodes)
#         logger.dump_tabular()
