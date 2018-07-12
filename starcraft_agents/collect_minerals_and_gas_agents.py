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
A collection of agents for collecting minerals.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import array as np_array
from numpy import linalg as np_linalg
from pysc2.agents import base_agents
from pysc2.lib import actions
from scipy import stats as sp_stats


class CollectMineralsAgent001(base_agents.ScriptedAgent):
    """
    A scripted agent for collecting minerals
    using a print out to determine a valid
    location of mineral fields.
    """
    def step(self, timestep):
        super(CollectMineralsAgent001, self).step(timestep)
        if timestep.observation['player'][self.idle_worker_count] > 0:
            if self.functions.Harvest_Gather_screen.id in timestep.observation['available_actions']:
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                mineralfields_y, mineralfields_x = (unit_type == self.neutral_mineralfields).nonzero()
                target_unit = [mineralfields_x[10], mineralfields_y[10]]
                return actions.FunctionCall(self.functions.Harvest_Gather_screen.id, [self.cmd_screen, target_unit])
            elif self.functions.select_idle_worker.id in timestep.observation['available_actions']:
                return actions.FunctionCall(self.functions.select_idle_worker.id, [self.select_worker_all])
        return actions.FunctionCall(self.functions.no_op.id, [])


class CollectMineralsAgent002(base_agents.ScriptedAgent):
    """
    A scripted agent for collecting minerals
    using the mode of the mineral fields to
    determine a valid location.
    """
    def step(self, timestep):
        super(CollectMineralsAgent002, self).step(timestep)
        if timestep.observation['player'][self.idle_worker_count] > 0:
            if self.functions.Harvest_Gather_screen.id in timestep.observation['available_actions']:
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                mineralfields_y, mineralfields_x = (unit_type == self.neutral_mineralfields).nonzero()
                mineralfield_x = sp_stats.mode(mineralfields_x)
                mineralfield_y = sp_stats.mode(mineralfields_y)
                target_unit = [mineralfield_x[0], mineralfield_y[0]]
                return actions.FunctionCall(self.functions.Harvest_Gather_screen.id, [self.cmd_screen, target_unit])
            elif self.functions.select_idle_worker.id in timestep.observation['available_actions']:
                return actions.FunctionCall(self.functions.select_idle_worker.id, [self.select_worker_all])
        return actions.FunctionCall(self.functions.no_op.id, [])


class CollectMineralsAgent003(base_agents.ScriptedAgent):
    """
    A simple agent for collecting minerals
    using the a basic search algorithm
    to find a valid mineral field.
    """
    def __init__(self):
        super(CollectMineralsAgent003, self).__init__()
        self.mineralfields_x = []
        self.mineralfields_y = []

    def step(self, timestep):
        super(CollectMineralsAgent003, self).step(timestep)
        if timestep.first():
            unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
            self.mineralfields_y, self.mineralfields_x = (unit_type == self.neutral_mineralfields).nonzero()
        if timestep.observation['player'][self.idle_worker_count] > 0:
            if self.functions.Harvest_Gather_screen.id in timestep.observation['available_actions']:
                target_unit = [self.mineralfields_x[self.steps], self.mineralfields_y[self.steps]]
                return actions.FunctionCall(self.functions.Harvest_Gather_screen.id, [self.cmd_screen, target_unit])
            elif self.functions.select_idle_worker.id in timestep.observation['available_actions']:
                return actions.FunctionCall(self.functions.select_idle_worker.id, [self.select_worker_all])
        return actions.FunctionCall(self.functions.no_op.id, [])


class CollectMineralsAgent004(base_agents.ScriptedAgent):
    """
    A simple agent for collecting minerals
    using the mean of either of the two
    mineral field clusters.
    """
    def __init__(self):
        super(CollectMineralsAgent004, self).__init__()
        self.greater_than_x = []
        self.greater_than_y = []
        self.less_than_x = []
        self.less_than_y = []
        self.mineralfields_x = []
        self.mineralfields_y = []

    def step(self, timestep):
        super(CollectMineralsAgent004, self).step(timestep)
        if timestep.first():
            unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
            self.mineralfields_y, self.mineralfields_x = (unit_type == self.neutral_mineralfields).nonzero()
            for x, y in zip(self.mineralfields_x, self.mineralfields_y):
                if x < 32:
                    self.less_than_x.append(x)
                    self.less_than_y.append(y)
                else:
                    self.greater_than_x.append(x)
                    self.greater_than_y.append(y)
        if timestep.observation['player'][self.idle_worker_count] > 0:
            if self.functions.Harvest_Gather_screen.id in timestep.observation['available_actions']:
                target_unit = [np_array(self.less_than_x[self.steps]), np_array(self.less_than_x[self.steps])]
                return actions.FunctionCall(self.functions.Harvest_Gather_screen.id, [self.cmd_screen, target_unit])
            elif self.functions.select_idle_worker.id in timestep.observation['available_actions']:
                return actions.FunctionCall(self.functions.select_idle_worker.id, [self.select_worker_all])
        return actions.FunctionCall(self.functions.no_op.id, [])


class CollectMineralsAgent005(base_agents.ScriptedAgent):

    def __init__(self):
        super(CollectMineralsAgent005, self).__init__()
        self.mineralfields_x = []
        self.mineralfields_y = []
        self.player_self = 1

    def step(self, timestep):
        super(CollectMineralsAgent005, self).step(timestep)
        if timestep.first():
            unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
            self.mineralfields_y, self.mineralfields_x = (unit_type == self.neutral_mineralfields).nonzero()
        if timestep.observation['player'][self.idle_worker_count] > 0:
            if self.functions.Harvest_Gather_screen.id in timestep.observation['available_actions']:
                selected = timestep.observation['screen'][self.screen_features.selected.index]
                player_y, player_x = (selected == self.player_self).nonzero()
                player = [int(player_x.mean()), int(player_y.mean())]
                index, min_dist = None, None
                for i in range(len(self.mineralfields_x)):
                    mf = [self.mineralfields_x[i], self.mineralfields_y[i]]
                    dist = np_linalg.norm(np_array(player) - np_array(mf))
                    if not min_dist or dist < min_dist:
                        index, min_dist = i, dist
                target_unit = [self.mineralfields_x[index + self.steps], self.mineralfields_y[index + self.steps]]
                return actions.FunctionCall(self.functions.Harvest_Gather_screen.id, [self.cmd_screen, target_unit])
            elif self.functions.select_idle_worker.id in timestep.observation['available_actions']:
                return actions.FunctionCall(self.functions.select_idle_worker.id, [self.select_worker_all])
        return actions.FunctionCall(self.functions.no_op.id, [])


class CollectMineralsAgent006(base_agents.ScriptedAgent):

    def __init__(self):
        super(CollectMineralsAgent006, self).__init__()
        self.mineralfields_x = []
        self.mineralfields_y = []

    def step(self, timestep):
        super(CollectMineralsAgent006, self).step(timestep)
        if timestep.first():
            unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
            self.mineralfields_y, self.mineralfields_x = (unit_type == self.neutral_mineralfields).nonzero()
        if timestep.observation['player'][self.idle_worker_count] > 0:
            if self.functions.Harvest_Gather_screen.id in timestep.observation['available_actions']:
                target_unit = [self.mineralfields_x[0], self.mineralfields_y[self.steps]]
                return actions.FunctionCall(self.functions.Harvest_Gather_screen.id, [self.cmd_screen, target_unit])
            elif self.functions.select_idle_worker.id in timestep.observation['available_actions']:
                return actions.FunctionCall(self.functions.select_idle_worker.id, [self.select_worker_all])
        return actions.FunctionCall(self.functions.no_op.id, [])


class CollectMineralsAgent007(base_agents.DeepQAgent):

    def step(self, timestep):
        super(CollectMineralsAgent007, self).step(timestep)
        screen = self.screen(timestep.observation)
        self.x_coord = self.act_x(np_array(screen)[None])[0]
        self.y_coord = self.act_y(np_array(screen)[None])[0]
        if self.functions.Move_screen.id in timestep.observation['available_actions']:
            return actions.FunctionCall(self.functions.Move_screen.id, [self.not_queued, [self.x_coord, self.y_coord]])
        elif self.functions.select_army.id in timestep.observation['available_actions']:
            return actions.FunctionCall(self.functions.select_army.id, [self.select_all])
        return actions.FunctionCall(self.functions.no_op.id, [])

    def training_step(self, timestep, **kwargs):
        super(CollectMineralsAgent007, self).step(timestep)
        screen = self.screen(timestep.observation)
        update_eps = kwargs.pop('update_eps')
        self.x_coord = self.act_x(np_array(screen)[None], update_eps, **kwargs)[0]
        self.y_coord = self.act_y(np_array(screen)[None], update_eps, **kwargs)[0]
        if self.functions.Move_screen.id in timestep.observation['available_actions']:
            return actions.FunctionCall(self.functions.Move_screen.id, [self.not_queued, [self.x_coord, self.y_coord]])
        elif self.functions.select_idle_worker.id in timestep.observation['available_actions']:
            return actions.FunctionCall(self.functions.select_idle_worker.id, [self.select_worker_all])
        return actions.FunctionCall(self.functions.no_op.id, [])


class CollectMineralsAndGasAgent001(base_agents.ScriptedAgent):

    def __init__(self):
        super(CollectMineralsAndGasAgent, self).__init__()
        self.refinery_count = 0

    def step(self, timestep):
        super(CollectMineralsAndGasAgent, self).step(timestep)
        if timestep.observation['player'][self.idle_worker_count] > 0:    # harvest minerals
            if self.functions.Harvest_Gather_screen.id in timestep.observation['available_actions']:
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                mineralfields_y, mineralfields_x = (unit_type == self.neutral_mineralfields).nonzero()
                target_unit = [mineralfields_x[10], mineralfields_y[10]]
                return actions.FunctionCall(self.functions.Harvest_Gather_screen.id, [self.cmd_screen, target_unit])
            else:    # select idle workers
                return actions.FunctionCall(self.functions.select_idle_worker.id, [self.select_worker_all])
        elif self.refinery_count == 0:    # build refinery
            if self.functions.Build_Refinery_screen.id in timestep.observation['available_actions']:
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                vespenegeysers_y, vespenegeysers_x = (unit_type == self.vespene_geyser).nonzero()
                target_unit = [vespenegeysers_x[10], vespenegeysers_y[10]]
                self.refinery_count += 1
                return actions.FunctionCall(self.functions.Build_Refinery_screen.id, [self.cmd_screen, target_unit])
            else:    # select worker
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                unit_y, unit_x = (unit_type == self.terran_scv).nonzero()
                target_unit = [unit_x[0], unit_y[0]]
                return actions.FunctionCall(self.functions.select_point.id, [self.cmd_screen, target_unit])
        return actions.FunctionCall(self.functions.no_op.id, [])


class CollectMineralsAndGasAgent002(base_agents.ScriptedAgent):

    def __init__(self):
        super(CollectMineralsAndGasAgent, self).__init__()
        self.refinery_count = 0

    def step(self, timestep):
        super(CollectMineralsAndGasAgent, self).step(timestep)
        if timestep.observation['player'][self.idle_worker_count] > 0:    # harvest minerals
            if self.functions.Harvest_Gather_screen.id in timestep.observation['available_actions']:
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                mineralfields_y, mineralfields_x = (unit_type == self.neutral_mineralfields).nonzero()
                target_unit = [mineralfields_x[10], mineralfields_y[10]]
                return actions.FunctionCall(self.functions.Harvest_Gather_screen.id, [self.cmd_screen, target_unit])
            else:    # select idle workers
                return actions.FunctionCall(self.functions.select_idle_worker.id, [self.select_worker_all])
        elif self.refinery_count == 0:    # build refinery
            if self.functions.Build_Refinery_screen.id in timestep.observation['available_actions']:
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                vespene_geysers_y, vespene_geysers_x = (unit_type == self.vespene_geyser).nonzero()
                vespene_geyser_x, vespene_geyser_y = [], []
                for x in vespene_geysers_x:
                    if x < 42:
                        vespene_geyser_x.append(x)
                for y in vespene_geysers_y:
                    if y < 42:
                        vespene_geyser_y.append(y)
                target_unit = [vespene_geyser_x[10], vespene_geyser_y[10]]
                self.refinery_count += 1
                return actions.FunctionCall(self.functions.Build_Refinery_screen.id, [self.cmd_screen, target_unit])
            else:    # select worker
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                unit_y, unit_x = (unit_type == self.terran_scv).nonzero()
                target_unit = [unit_x[0], unit_y[0]]
                return actions.FunctionCall(self.functions.select_point.id, [self.cmd_screen, target_unit])
        return actions.FunctionCall(self.functions.no_op.id, [])
