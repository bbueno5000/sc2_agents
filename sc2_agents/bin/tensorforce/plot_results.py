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
Plot the data gathered from StarCraft II experiments.
"""
from collections import defaultdict
from dill import load as dill_load
from matplotlib import pyplot
from numpy import add as np_add
from numpy import array as np_array
from numpy import cumsum as np_cumsum
from numpy import digitize as np_digitize
from numpy import float32 as np_float32    # pylint: disable=E0611
from numpy import linspace as np_linspace
from numpy import where as np_where
from numpy import zeros as np_zeros
from pandas import DataFrame as pd_DataFrame
from pandas import Series as pd_Series
from seaborn import tsplot


MAX_TSTEPS = int(2e3)
NSAMPLES = 100


def main():
    with open("pysc2/data/results.pkl", 'rb') as file:
        results = dill_load(file)
    experiment_labels = {'starcraft-a': "Double Q Learning",
                         'starcraft-duel-a': "Dueling Double Q Learning",
                         'starcraft-prior-a': "Double Q Learning with Prioritized Replay",
                         'starcraft-prior-duel-a': "Dueling Double Q Learning with Prioritized Replay"}
    experiments = experiment_labels.keys()
    agent_data = defaultdict(lambda: defaultdict(lambda: []))
    for experiment in experiments:
        for name, data in results.items():
            if name.startswith(experiment):
                agent = data['agent_id']
                times = np_cumsum(data['episode_data']['episode_lengths'])
                rewards = np_array(data['episode_data']['episode_rewards'])
                rewards = rewards[times < MAX_TSTEPS]
                times = times[times < MAX_TSTEPS]
                agent_data[agent][experiment].append((times, rewards))
    experiments_to_plot = ['starcraft-a']
    plot_experiments(experiment_labels, experiments_to_plot, agent_data)


def plot_experiments(experiment_name, experiments_to_plot, agent_data, ncols=4):
    """
    Given a set of experiments, plot them in a line graph.
    experiments_to_plot: Chosen experiments to plot.
    """
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
    assert len(colors) >= len(experiments_to_plot)
    color_legend = dict(zip(experiments_to_plot, colors))
    # Print the legend
    for experiment, color in color_legend.items():
        print(color, ":", experiment_name[experiment])
    # Select relevant games for those experiments
    all_games = list(agent_data.keys())
    relevant_agents = [g for g in all_games if any(e in agent_data[g] for e in experiments_to_plot)]
    # Create the figure
    # TODO: fix figures columns and rows
    # ncols = min(ncols, len(relevant_agents))
    # nrows = (len(relevant_agents) + ncols - 1) // ncols
    ncols = 2
    nrows = 2
    _, axes = pyplot.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1:
        axes = [axes]
    # Plot the data
    for index, agent in enumerate(sorted(relevant_agents)):
        ax = axes[index // ncols][index % ncols]
        ax.set_title(agent)
        for experiment_label, experiment_data in agent_data[agent].items():
            if experiment_label in experiments_to_plot:
                tsplot(ax=ax,
                       ci=[68, 95],
                       color=color_legend[experiment_label],
                       data=translate_episode_data(experiment_data),
                       time="Frame",
                       unit="run_id",
                       value="Average Episode Reward")
    pyplot.show()


def sample(bins, time, value):
    """
    Given value[i] was observed at time[i],
    group them into bins i.e.,
    *(bins[j], bins[j+1], ...)*

    Values for bin j are equal to the
    average of all value[k] and,
    bin[j] <= time[k] < bin[j+1].

    __Arguments__
    bins: _np.array_
        Endpoints of the bins.
        For n bins it shall be of length n + 1.
    t: _np.array_
        Times at which the values are observed.
    vt: _np.array_
        Values for those times.

    __Returns__
    x: _np.array_
        Endspoints of all the bins.
    y: _np.array_
        Average values in all bins.
    """
    bin_idx = np_digitize(time, bins) - 1
    value_sums = np_zeros(shape=len(bins) - 1, dtype=np_float32)
    value_cnts = np_zeros(shape=len(bins) - 1, dtype=np_float32)
    np_add.at(value_sums, bin_idx, value)
    np_add.at(value_cnts, bin_idx, 1)
    # ensure graph has no holes
    zeros = np_where(value_cnts == 0)
    assert value_cnts[0] > 0
    for z in zeros:
        value_sums[z] = value_sums[z - 1]
        value_cnts[z] = value_cnts[z - 1]
    return bins[1:], value_sums / value_cnts


def translate_episode_data(episode_data):
    """
    Convert episode data into data that
    can be used in a graph.

    Given data from multiple episodes make
    it such that it can be plotted by tsplot,
    i.e. the mean plus the confidence bounds.
    """
    times, units, values = [], [], []
    for index, (ep_len, ep_rew) in enumerate(episode_data):
        # Smooth out the data
        ep_rew = pd_Series(ep_rew).ewm(span=1000).mean()
        # sample for faster plotting
        x, y = sample(bins=np_linspace(0, MAX_TSTEPS, NSAMPLES + 1),
                      time=ep_len,
                      value=ep_rew)
        # Convert to tsplot format
        times.extend(x)
        values.extend(y)
        units.extend([index] * len(x))
    return pd_DataFrame({'Frame': times, 'run_id': units, 'Average Episode Reward': values})


if __name__ == "__main__":
    main()
