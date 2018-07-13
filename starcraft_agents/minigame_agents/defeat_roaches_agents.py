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
A collection of agents for defeating roaches
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import argmax as np_argmax
from pysc2.agents import scripted_agent
from pysc2.lib import actions
from pysc2.lib import features


class DefeatRoachesAgent(scripted_agent.DefeatRoaches):
    """
    Generic agent for defeating roachez
    """

    def __init__(self):
        super(DefeatRoachesAgent, self).__init__()
        self.functions = actions.FUNCTIONS
        self.screen_features = features.SCREEN_FEATURES
        self.cmd_screen = [0]
        self.idle_worker_count = 7
        self.neutral_mineralfields = 341
        self.not_queued = [0]
        self.player_friendly = 1
        self.player_neutral = 3    # beacon/minerals
        self.results = {}
        self.results['agent_id'] = self.__class__.__name__
        self.results['episode_data'] = {'episode_lengths': [], 'episode_rewards': []}
        self.select_all = [0]
        self.select_worker_all = [2]
        self.terran_commandcenter = 18
        self.terran_scv = 45
        self.vespene_geyser = 342

    def reset(self):
        super(DefeatRoachesAgent, self).reset()
        self.mean_reward = 0
        self.results['episode_data']['episode_lengths'].append(self.steps)
        self.results['episode_data']['episode_rewards'].append(self.reward)
        self.reward = 0
        self.steps = 0


class DefeatRoachesAgent001(DefeatRoachesAgent):
    """
    Basic agent for defeating roaches
    """

    def __init__(self):
        super(DefeatRoachesAgent001, self).__init__()
        self.player_hostile = 4

    def step(self, timestep):
        super(DefeatRoachesAgent001, self).step(timestep)
        if self.functions.Attack_screen.id in timestep.observation.available_actions:
            player_relative = timestep.observation.feature_screen.player_relative
            hostiles_y, hostiles_x = (player_relative == self.player_hostile).nonzero()
            if not hostiles_y.any():
                return actions.FunctionCall(self.functions.no_op.id, [])
            index = np_argmax(hostiles_y)
            target_unit = [hostiles_x[index], hostiles_y[index]]
            return actions.FunctionCall(self.functions.Attack_screen.id, [self.not_queued, target_unit])
        elif self.functions.select_army.id in timestep.observation.available_actions:
            return actions.FunctionCall(self.functions.select_army.id, [self.select_all])
        return actions.FunctionCall(self.functions.no_op.id, [])
