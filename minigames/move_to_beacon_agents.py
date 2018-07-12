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
A collection of agents for moving to a beacon
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import array as np_array
from pysc2.agents import scripted_agent
from pysc2.lib import actions
from pysc2.lib import features


class MoveToBeaconAgent(scripted_agent.MoveToBeacon):
    """
    Generic agent for moving to a beacon
    """

    def __init__(self):
        super(MoveToBeaconAgent, self).__init__()
        self.functions = actions.FUNCTIONS
        self.screen_features = features.SCREEN_FEATURES
        self.results = {}
        self.results['agent_id'] = self.__class__.__name__
        self.results['episode_data'] = {'episode_lengths': [], 'episode_rewards': []}

    def reset(self):
        super(MoveToBeaconAgent, self).reset()
        self.mean_reward = 0
        self.results['episode_data']['episode_lengths'].append(self.steps)
        self.results['episode_data']['episode_rewards'].append(self.reward)
        self.reward = 0
        self.steps = 0


class MoveToBeaconAgent001(MoveToBeaconAgent):
    """
    Basic agent for moving to a beacon
    """

    def __init__(self):
        super(MoveToBeaconAgent001, self).__init__()
        self.cmd_screen = [0]
        self.not_queued = [0]
        self.player_friendly = 1
        self.player_neutral = 3    # beacon/minerals
        self.select_all = [0]

    def step(self, timestep):
        if self.functions.Move_screen.id in timestep.observation.available_actions:
            player_relative = timestep.observation.feature_screen.player_relative
            neutral_y, neutral_x = (player_relative == self.player_neutral).nonzero()
            if not neutral_y.any():
                return actions.FunctionCall(self.functions.no_op.id, [])
            target = [int(neutral_x.mean()), int(neutral_y.mean())]
            return actions.FunctionCall(self.functions.Move_screen.id, [self.not_queued, target])
        else:
            return actions.FunctionCall(self.functions.select_army.id, [self.select_all])
        return actions.FunctionCall(self.functions.no_op.id, [])


class MoveToBeaconAgent002(MoveToBeaconAgent):
    """
    DeepQ agent for moving to a beacon
    """

    def __init__(self, act_x, act_y):
        super(MoveToBeaconAgent002, self).__init__()
        self.act_x = act_x
        self.act_y = act_y
        self.mean_reward = 0
        self.x_coord = 0
        self.y_coord = 0
        self.not_queued = [0]
        self.player_neutral = 3
        self.select_all = [0]
        self.select_worker_all = [2]

    def screen(self, obs):
        player_relative = obs.observation.feature_screen.player_relative
        return (player_relative == self.player_neutral).astype(int)

    def step(self, timestep):
        super(MoveToBeaconAgent002, self).step(timestep)
        screen = self.screen(timestep.observation)
        self.x_coord = self.act_x(np_array(screen)[None])[0]
        self.y_coord = self.act_y(np_array(screen)[None])[0]
        if self.functions.Move_screen.id in timestep.observation.available_actions:
            return actions.FunctionCall(self.functions.Move_screen.id, [self.not_queued, [self.x_coord, self.y_coord]])
        elif self.functions.select_army.id in timestep.observation.available_actions:
            return actions.FunctionCall(self.functions.select_army.id, [self.select_all])
        return actions.FunctionCall(self.functions.no_op.id, [])

    def training_step(self, timestep, **kwargs):
        super(MoveToBeaconAgent002, self).step(timestep)
        screen = self.screen(timestep.observation)
        update_eps = kwargs.pop('update_eps', "key not found")
        self.x_coord = self.act_x(np_array(screen)[None], update_eps, **kwargs)[0]
        self.y_coord = self.act_y(np_array(screen)[None], update_eps, **kwargs)[0]
        if self.functions.Move_screen.id in timestep.observation.available_actions:
            return actions.FunctionCall(self.functions.Move_screen.id, [self.not_queued, [self.x_coord, self.y_coord]])
        elif self.functions.select_army.id in timestep.observation.available_actions:
            return actions.FunctionCall(self.functions.select_army.id, [self.select_all])
        return actions.FunctionCall(self.functions.no_op.id, [])
