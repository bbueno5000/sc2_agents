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
A collection of agents for collecting mineral shards
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from baselines import logger
from numpy import array as np_array
from numpy import linalg as np_linalg
from pysc2.agents import scripted_agent
from pysc2.lib import actions
from pysc2.lib import features


class CollectMineralShardsAgent(scripted_agent.CollectMineralShards):
    """
    Generic agent for collecting mineral shards
    """

    def __init__(self):
        super(CollectMineralShardsAgent, self).reset()
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
        super(CollectMineralShardsAgent, self).reset()
        self.mean_reward = 0
        self.results['episode_data']['episode_lengths'].append(self.steps)
        self.results['episode_data']['episode_rewards'].append(self.reward)
        self.reward = 0
        self.steps = 0
        logger.record_tabular("episodes", self.episodes)
        logger.dump_tabular()


class CollectMineralShardsAgent001(CollectMineralShardsAgent):
    """
    Basic agent for collecting mineral shards
    """

    def step(self, timestep):
        super(CollectMineralShardsAgent001, self).step(timestep)
        if self.functions.Move_screen.id in timestep.observation.available_actions:
            player_relative = timestep.observation.feature_screen.player_relative
            neutral_y, neutral_x = (player_relative == self.player_neutral).nonzero()
            player_y, player_x = (player_relative == self.player_friendly).nonzero()
            player = [int(player_x.mean()), int(player_y.mean())]
            closest, min_dist = None, None
            for p in zip(neutral_x, neutral_y):
                dist = np_linalg.norm(np_array(player) - np_array(p))
                if not min_dist or dist < min_dist:
                    closest, min_dist = p, dist
            return actions.FunctionCall(self.functions.Move_screen.id, [self.not_queued, closest])
        else:
            return actions.FunctionCall(self.functions.select_army.id, [self.select_all])
        return actions.FunctionCall(self.functions.no_op.id, [])


class CollectMineralShardsAgent002(CollectMineralShardsAgent):
    """
    DeepQ agent for collecting mineral shards
    """

    def __init__(self, act_x, act_y):
        super(CollectMineralShardsAgent002, self).__init__()
        self.functions = actions.FUNCTIONS
        self.screen_features = features.SCREEN_FEATURES
        self.act_x = act_x
        self.act_y = act_y
        self.mean_reward = 0
        self.x_coord = 0
        self.y_coord = 0
        self.not_queued = [0]
        self.player_neutral = 3
        self.select_all = [0]
        self.select_worker_all = [2]

    def screen(self, timestep):
        player_relative = timestep.observation.feature_screen.player_relative
        return (player_relative == self.player_neutral).astype(int)

    def step(self, timestep):
        super(CollectMineralShardsAgent002, self).step(timestep)
        screen = self.screen(timestep.observation)
        self.x_coord = self.act_x(np_array(screen)[None])[0]
        self.y_coord = self.act_y(np_array(screen)[None])[0]
        if self.functions.Move_screen.id in timestep.observation.available_actions:
            return actions.FunctionCall(self.functions.Move_screen.id, [self.not_queued, [self.x_coord, self.y_coord]])
        elif self.functions.select_army.id in timestep.observation.available_actions:
            return actions.FunctionCall(self.functions.select_army.id, [self.select_all])
        return actions.FunctionCall(self.functions.no_op.id, [])

    def training_step(self, timestep, **kwargs):
        super(CollectMineralShardsAgent002, self).step(timestep)
        screen = self.screen(timestep.observation)
        update_eps = kwargs.pop('update_eps', "Key not found.")
        self.x_coord = self.act_x(np_array(screen)[None], update_eps, **kwargs)[0]
        self.y_coord = self.act_y(np_array(screen)[None], update_eps, **kwargs)[0]
        if self.functions.Move_screen.id in timestep.observation.available_actions:
            return actions.FunctionCall(self.functions.Move_screen.id, [self.not_queued, [self.x_coord, self.y_coord]])
        elif self.functions.select_army.id in timestep.observation.available_actions:
            return actions.FunctionCall(self.functions.select_army.id, [self.select_all])
        return actions.FunctionCall(self.functions.no_op.id, [])
