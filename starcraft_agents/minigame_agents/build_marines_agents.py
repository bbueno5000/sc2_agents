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
A collection of agents related to building marines.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions


class BuildBarracksAgent(BaseAgent):
    """
    Generic agent for building barracks.
    """

    def __init__(self):
        super(BuildBarracksAgent, self).reset()
        self.barracks_count = 0
        self.cmd_screen = [0]
        self.idle_worker_count = 7
        self.functions = actions.FUNCTIONS
        self.neutral_mineralfields = 341
        self.not_queued = [0]
        self.player_friendly = 1
        self.player_neutral = 3    # beacon/minerals
        self.results = {}
        self.results['agent_id'] = self.__class__.__name__
        self.results['episode_data'] = {'episode_lengths': [], 'episode_rewards': []}
        self.select_all = [0]
        self.select_worker_all = [2]
        self.supply_depot_count = 0
        self.terran_barrack_id = 21
        self.terran_commandcenter = 18
        self.terran_scv = 45
        self.vespene_geyser = 342

    def reset(self):
        super(BuildBarracksAgent, self).reset()
        self.mean_reward = 0
        self.results['episode_data']['episode_lengths'].append(self.steps)
        self.results['episode_data']['episode_rewards'].append(self.reward)
        self.reward = 0
        self.steps = 0


class BuildBarracksAgent001(BuildBarracksAgent):
    """
    Scripted agent for building barracks.
    """

    def step(self, timestep):
        super(BuildBarracksAgent001, self).step(timestep)
        if self.supply_depot_count == 0:    # build supply depot
            if self.functions.Build_SupplyDepot_screen.id in timestep.observation.available_actions:
                unit_type = timestep.observation.feature_screen.unit_type
                self.cmdcenters_y, self.cmdcenters_x = (unit_type == self.terran_commandcenter).nonzero()
                target_point = [int(self.cmdcenters_x.mean()), int(self.cmdcenters_y.mean()) - 15]
                self.supply_depot_count += 1
                return actions.FunctionCall(self.functions.Build_SupplyDepot_screen.id, [self.cmd_screen, target_point])
            else:     # select scv
                unit_type = timestep.observation.feature_screen.unit_type
                scvs_y, scvs_x = (unit_type == self.terran_scv).nonzero()
                target_unit = [scvs_x[0], scvs_y[0]]
                return actions.FunctionCall(self.functions.select_point.id, [self.cmd_screen, target_unit])
        if self.barracks_count == 0:    # build barracks
            if self.functions.Build_Barracks_screen.id in timestep.observation.available_actions:
                target_point = [int(self.cmdcenters_x.mean()) + 20, int(self.cmdcenters_y.mean())]
                self.barracks_count += 1
                return actions.FunctionCall(self.functions.Build_Barracks_screen.id, [self.cmd_screen, target_point])
        return actions.FunctionCall(self.functions.no_op.id, [])


class BuildMarinesAgent(BaseAgent):
    """
    Generic agent for building marines.
    """

    def __init__(self):
        super(BuildMarinesAgent, self).reset()
        self.barracks_count = 0
        self.functions = actions.FUNCTIONS
        self.cmd_screen = [0]
        self.idle_worker_count = 7
        self.neutral_mineralfields = 341
        self.not_queued = [0]
        self.player_friendly = 1
        self.player_neutral = 3    # beacon/minerals
        self.queued = [1]
        self.results = {}
        self.results['agent_id'] = self.__class__.__name__
        self.results['episode_data'] = {'episode_lengths': [], 'episode_rewards': []}
        self.select_all = [0]
        self.select_worker_all = [2]
        self.supply_depot_count = 0
        self.supply_max_id = 4
        self.supply_used_id = 3
        self.terran_barrack_id = 21
        self.terran_commandcenter = 18
        self.terran_scv = 45
        self.vespene_geyser = 342

    def reset(self):
        super(BuildMarinesAgent, self).reset()
        self.results['episode_data']['episode_lengths'].append(self.steps)
        self.results['episode_data']['episode_rewards'].append(self.reward)
        self.reward = 0
        self.steps = 0


class BuildMarinesAgent001(BuildMarinesAgent):
    """
    Scripted agent for building marines.
    """

    def step(self, timestep):
        super(BuildMarinesAgent001, self).step(timestep)
        if self.supply_depot_count == 0:    # build supply depot
            if self.functions.Build_SupplyDepot_screen.id in timestep.observation.available_actions:
                unit_type = timestep.observation.feature_screen.unit_type
                self.cmdcenters_y, self.cmdcenters_x = (unit_type == self.terran_commandcenter).nonzero()
                target_point = [int(self.cmdcenters_x.mean()), int(self.cmdcenters_y.mean()) - 15]
                self.supply_depot_count += 1
                return actions.FunctionCall(self.functions.Build_SupplyDepot_screen.id, [self.cmd_screen, target_point])
            else:     # select scv
                unit_type = timestep.observation.feature_screen.unit_type
                scvs_y, scvs_x = (unit_type == self.terran_scv).nonzero()
                target_unit = [scvs_x[0], scvs_y[0]]
                return actions.FunctionCall(self.functions.select_point.id, [self.cmd_screen, target_unit])
        if self.barracks_count == 0:    # build barracks
            if self.functions.Build_Barracks_screen.id in timestep.observation.available_actions:
                target_point = [int(self.cmdcenters_x.mean()) + 20, int(self.cmdcenters_y.mean())]
                self.barracks_count += 1
                return actions.FunctionCall(self.functions.Build_Barracks_screen.id, [self.cmd_screen, target_point])
        if self.functions.Train_Marine_quick.id in timestep.observation.available_actions:    # train marines
            if timestep.observation['player'][self.supply_used_id] < timestep.observation['player'][self.supply_max_id]:
                return actions.FunctionCall(self.functions.Train_Marine_quick.id, [self.queued])
        else:    # select barracks
            unit_type = timestep.observation.feature_screen.unit_type
            barracks_y, barracks_x = (unit_type == self.terran_barrack_id).nonzero()
            if not barracks_y.any():
                return actions.FunctionCall(self.functions.no_op.id, [])
            target_point = [int(barracks_x.mean()), int(barracks_y.mean())]
            return actions.FunctionCall(self.functions.select_point.id, [self.cmd_screen, target_point])
        return actions.FunctionCall(self.functions.no_op.id, [])


class BuildSupplyDepotAgent(BaseAgent):
    """
    Generic agent for building supply depot.
    """

    def __init__(self):
        super(BuildSupplyDepotAgent, self).reset()
        self.functions = actions.FUNCTIONS
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
        self.supply_depot_count = 0
        self.terran_commandcenter = 18
        self.terran_scv = 45
        self.vespene_geyser = 342

    def reset(self):
        super(BuildSupplyDepotAgent, self).reset()
        self.mean_reward = 0
        self.results['episode_data']['episode_lengths'].append(self.steps)
        self.results['episode_data']['episode_rewards'].append(self.reward)
        self.reward = 0
        self.steps = 0


class BuildSupplyDepotAgent001(BuildBarracksAgent):
    """
    Scripted agent for building supply depot.
    """

    def step(self, timestep):
        super(BuildSupplyDepotAgent001, self).step(timestep)
        if self.supply_depot_count == 0:    # build supply depot
            if self.functions.Build_SupplyDepot_screen.id in timestep.observation.available_actions:
                unit_type = timestep.observation.feature_screen.unit_type
                cmdcenters_y, cmdcenters_x = (unit_type == self.terran_commandcenter).nonzero()
                target_point = [int(cmdcenters_x.mean()), int(cmdcenters_y.mean()) - 15]
                self.supply_depot_count += 1
                return actions.FunctionCall(self.functions.Build_SupplyDepot_screen.id, [self.cmd_screen, target_point])
            else:   # select scv
                unit_type = timestep.observation.feature_screen.unit_type
                scvs_y, scvs_x = (unit_type == self.terran_scv).nonzero()
                target_unit = [scvs_x[0], scvs_y[0]]
                return actions.FunctionCall(self.functions.select_point.id, [self.cmd_screen, target_unit])
        return actions.FunctionCall(self.functions.no_op.id, [])
