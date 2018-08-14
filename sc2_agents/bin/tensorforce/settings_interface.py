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

from absl import app
from absl import flags
from dqn_agent import train_agent
from pysc2.bin.agent import run_thread
from tkinter import Button
from tkinter import Entry
from tkinter import Label
from tkinter import StringVar
from tkinter import OptionMenu
from tkinter import Tk

FLAGS = flags.FLAGS

class _UserInterface:

    def __init__(self):
        self.agent_names = [
            "starcraft_agents.minigame_agents.collect_mineral_shards_agents.CollectMineralShardsAgent",
            "starcraft_agents.minigame_agents.collect_minerals_agents.CollectMineralsAgent",
            "starcraft_agents.minigame_agents.move_to_beacon_agents.MoveToBeaconAgent"]
        self.master = Tk()
        self.master.geometry('400x400')
        self.pady = 5
        self.switch = ["True", "False"]
        self.width = 25
        self._map_name()
        self._num_episodes()
        self._step_mul()
        self._save_replay()
        self._visualize()
        self._button()

    def __button_pressed(self):
        self.master.destroy()

    def __set_map(self, map_name):
        self.map_name = map_name

    def __set_save_replay(self, save_replay):
        self.save_replay = save_replay

    def __set_visualize(self, visualize):
        self.visualize = visualize

    def _button(self):
        button = Button(self.master,
                        text="Enter",
                        command=self.__button_pressed,
                        width=self.width)
        button.grid(columnspan=2, pady=20, row=11)

    def _map_name(self):
        self.map_names = ["BuildMarines",
                          "CollectMineralsAndGas",
                          "CollectMineralShards",
                          "DefeatRoaches",
                          "DefeatZerglingsAndBanelings",
                          "FindAndDefeatZerglings",
                          "MoveToBeacon"]
        self.map_name = self.map_names[0]
        string_var = StringVar(self.master, self.map_name)
        label = Label(self.master, text="map_name", width=self.width)
        options_menu = OptionMenu(self.master,
                                  string_var,
                                  *self.map_names,
                                  command=self.__set_map)
        label.grid(column=0, row=0)
        options_menu.grid(column=0, pady=self.pady, row=1)

    def _num_episodes(self):
        self.num_episodes = 10
        string_var = StringVar(self.master, self.num_episodes)
        label = Label(self.master, text="num_episodes", width=self.width)
        entry = Entry(self.master,
                      justify='center',
                      text=string_var.get(),
                      textvariable=string_var,
                      width=self.width)
        label.grid(column=0, row=4)
        entry.grid(column=0, pady=self.pady, row=5)

    def _save_replay(self):
        self.save_replay = False
        string_var = StringVar(self.master, self.save_replay)
        label = Label(self.master, text="save_replay", width=self.width)
        options_menu = OptionMenu(self.master,
                                  string_var,
                                  *self.switch,
                                  command=self.__set_save_replay)
        label.grid(column=1, row=8)
        options_menu.grid(column=1, pady=self.pady, row=9)

    def _step_mul(self):
        self.step_mul = 8
        string_var = StringVar(self.master, self.step_mul)
        label = Label(self.master, text="step_mul", width=self.width)
        entry = Entry(self.master,
                      justify='center',
                      text=string_var.get(),
                      textvariable=string_var,
                      width=self.width)
        label.grid(column=0, row=6)
        entry.grid(column=0, pady=self.pady, row=7)

    def _visualize(self):
        self.visualize = False
        string_var = StringVar(self.master, self.visualize)
        label = Label(self.master, text="visualize", width=self.width)
        options_menu = OptionMenu(self.master,
                                  string_var,
                                  *self.switch,
                                  command=self.__set_visualize)
        label.grid(column=0, row=8)
        options_menu.grid(column=0, pady=self.pady, row=9)

class RunAgent(_UserInterface):

    """
    GUI for selecting from the various options when running an agent.
    """

    def __init__(self):
        super(RunAgent, self).__init__()
        self.master.title("DeepQ Training")
        self._experiment_num()

    def __button_pressed(self):
        super(RunAgent, self).__init__()
        FLAGS.agent_name = self.agent_names[self.map_names.index(self.map_name)]
        FLAGS.experiment_num = self.experiment_num
        FLAGS.map_name = self.map_name
        FLAGS.num_episodes = self.num_episodes
        FLAGS.save_replay = self.save_replay
        FLAGS.step_mul = self.step_mul
        FLAGS.visualize = self.visualize
        app.run(run_thread)

    def _experiment_num(self):
        self.experiment_num = 1
        string_var = StringVar(self.master, self.experiment_num)
        label = Label(self.master, text="experiment_num", width=self.width)
        entry = Entry(self.master,
                      justify='center',
                      text=string_var.get(),
                      textvariable=string_var,
                      width=self.width)
        label.grid(column=0, row=2)
        entry.grid(column=0, pady=self.pady, row=3)

class TrainAgent(_UserInterface):

    """
    GUI for selecting from the various options when training a DeepQ agent.
    """

    def __init__(self):
        super(TrainAgent, self).__init__()
        self.master.title("DeepQ Training")
        self._convs()
        self._hiddens()
        self._dueling()
        self._prioritized_replay()

    def __button_pressed(self):
        super(TrainAgent, self).__init__()
        FLAGS.agent_name = self.agent_names[self.map_names.index(self.map_name)]
        FLAGS.convs = self.convs
        FLAGS.dueling = self.dueling
        FLAGS.hiddens = self.hiddens
        FLAGS.map_name = self.map_name
        FLAGS.num_episodes = self.num_episodes
        FLAGS.prioritized_replay = self.prioritized_replay
        FLAGS.save_replay = self.save_replay
        FLAGS.step_mul = self.step_mul
        FLAGS.visualize = self.visualize
        app.run(train_agent)

    def __set_dueling(self, dueling):
        self.dueling = dueling

    def __set_prioritized_replay(self, prioritized_replay):
        self.prioritized_replay = prioritized_replay

    def _convs(self):
        self.convs = ((8, 16, 4), (4, 32, 2))
        string_var = StringVar(self.master, self.convs)
        label = Label(self.master, text="convs", width=self.width)
        entry = Entry(self.master,
                      justify='center',
                      text=string_var.get(),
                      textvariable=string_var,
                      width=self.width)
        label.grid(column=1, row=0)
        entry.grid(column=1, pady=self.pady, row=1)

    def _dueling(self):
        self.dueling = True
        string_var = StringVar(self.master, self.dueling)
        label = Label(self.master, text="dueling", width=self.width)
        options_menu = OptionMenu(self.master,
                                  string_var,
                                  *self.switch,
                                  command=self.__set_dueling)
        label.grid(column=1, row=4)
        options_menu.grid(column=1, pady=self.pady, row=5)

    def _hiddens(self):
        self.hiddens = (125)
        string_var = StringVar(self.master, self.hiddens)
        label = Label(self.master, text="hiddens", width=self.width)
        entry = Entry(self.master,
                      justify='center',
                      text=string_var.get(),
                      textvariable=string_var,
                      width=self.width)
        label.grid(column=1, row=2)
        entry.grid(column=1, pady=self.pady, row=3)

    def _prioritized_replay(self):
        self.prioritized_replay = False
        string_var = StringVar(self.master, self.prioritized_replay)
        label = Label(self.master, text="prioritized_replay", width=self.width)
        options_menu = OptionMenu(self.master,
                                  string_var,
                                  *self.switch,
                                  command=self.__set_prioritized_replay)
        label.grid(column=1, row=6)
        options_menu.grid(column=1, pady=self.pady, row=7)
