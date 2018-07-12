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
A collection of agents pertaining to building marines
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pysc2.lib import actions
from pysc2.lib import features


class BuildBarracksAgent(base_agents.ScriptedAgent):

    def __init__(self):
        super(BuildBarracksAgent, self).__init__()
        self.barracks_count = 0
        self.supply_depot_count = 0
        self.terran_barrack_id = 21

    def step(self, timestep):
        super(BuildBarracksAgent, self).step(timestep)
        if self.supply_depot_count == 0:    # build supply depot
            if self.functions.Build_SupplyDepot_screen.id in timestep.observation['available_actions']:
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                self.commandcenters_y, self.commandcenters_x = (unit_type == self.terran_commandcenter).nonzero()
                target_point = [int(self.commandcenters_x.mean()), int(self.commandcenters_y.mean()) - 15]
                self.supply_depot_count += 1
                return actions.FunctionCall(self.functions.Build_SupplyDepot_screen.id, [self.cmd_screen, target_point])
            else:     # select scv
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                scvs_y, scvs_x = (unit_type == self.terran_scv).nonzero()
                target_unit = [scvs_x[0], scvs_y[0]]
                return actions.FunctionCall(self.functions.select_point.id, [self.cmd_screen, target_unit])
        if self.barracks_count == 0:    # build barracks
            if self.functions.Build_Barracks_screen.id in timestep.observation['available_actions']:
                target_point = [int(self.commandcenters_x.mean()) + 20, int(self.commandcenters_y.mean())]
                self.barracks_count += 1
                return actions.FunctionCall(self.functions.Build_Barracks_screen.id, [self.cmd_screen, target_point])
        return actions.FunctionCall(self.functions.no_op.id, [])


class BuildMarinesAgent(base_agents.ScriptedAgent):

    def __init__(self):
        super(BuildMarinesAgent, self).__init__()
        self.barracks_count = 0
        self.queued = [1]
        self.supply_depot_count = 0
        self.supply_max_id = 4
        self.supply_used_id = 3
        self.terran_barrack_id = 21

    def step(self, timestep):
        super(BuildMarinesAgent, self).step(timestep)
        if self.supply_depot_count == 0:    # build supply depot
            if self.functions.Build_SupplyDepot_screen.id in timestep.observation['available_actions']:
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                self.commandcenters_y, self.commandcenters_x = (unit_type == self.terran_commandcenter).nonzero()
                target_point = [int(self.commandcenters_x.mean()), int(self.commandcenters_y.mean()) - 15]
                self.supply_depot_count += 1
                return actions.FunctionCall(self.functions.Build_SupplyDepot_screen.id, [self.cmd_screen, target_point])
            else:     # select scv
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                scvs_y, scvs_x = (unit_type == self.terran_scv).nonzero()
                target_unit = [scvs_x[0], scvs_y[0]]
                return actions.FunctionCall(self.functions.select_point.id, [self.cmd_screen, target_unit])
        if self.barracks_count == 0:    # build barracks
            if self.functions.Build_Barracks_screen.id in timestep.observation['available_actions']:
                target_point = [int(self.commandcenters_x.mean()) + 20, int(self.commandcenters_y.mean())]
                self.barracks_count += 1
                return actions.FunctionCall(self.functions.Build_Barracks_screen.id, [self.cmd_screen, target_point])
        if self.functions.Train_Marine_quick.id in timestep.observation['available_actions']:    # train marines
            if timestep.observation['player'][self.supply_used_id] < timestep.observation['player'][self.supply_max_id]:
                return actions.FunctionCall(self.functions.Train_Marine_quick.id, [self.queued])
        else:    # select barracks
            unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
            barracks_y, barracks_x = (unit_type == self.terran_barrack_id).nonzero()
            if not barracks_y.any():
                return actions.FunctionCall(self.functions.no_op.id, [])
            target_point = [int(barracks_x.mean()), int(barracks_y.mean())]
            return actions.FunctionCall(self.functions.select_point.id, [self.cmd_screen, target_point])
        return actions.FunctionCall(self.functions.no_op.id, [])


class BuildSupplyDepotAgent(base_agents.ScriptedAgent):

    def __init__(self):
        super(BuildSupplyDepotAgent, self).__init__()
        self.supply_depot_count = 0

    def step(self, timestep):
        super(BuildSupplyDepotAgent, self).step(timestep)
        if self.supply_depot_count == 0:    # build supply depot
            if self.functions.Build_SupplyDepot_screen.id in timestep.observation['available_actions']:
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                commandcenters_y, commandcenters_x = (unit_type == self.terran_commandcenter).nonzero()
                target_point = [int(commandcenters_x.mean()), int(commandcenters_y.mean()) - 15]
                self.supply_depot_count += 1
                return actions.FunctionCall(self.functions.Build_SupplyDepot_screen.id, [self.cmd_screen, target_point])
            else:   # select scv
                unit_type = timestep.observation['screen'][self.screen_features.unit_type.index]
                scvs_y, scvs_x = (unit_type == self.terran_scv).nonzero()
                target_unit = [scvs_x[0], scvs_y[0]]
                return actions.FunctionCall(self.functions.select_point.id, [self.cmd_screen, target_unit])
        return actions.FunctionCall(self.functions.no_op.id, [])
