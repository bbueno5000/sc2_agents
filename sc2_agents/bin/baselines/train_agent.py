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
from baselines import deepq
from gym import make
from gym_sc2 import envs

def train_deepq_agent(env, network):
    act = deepq.learn(
        env,
        network,
        total_timesteps=1000)
    act.save('./move_to_beacon')

def train_ppo_agent(env, network):
    act = deepq.learn(
        env,
        network,
        total_timesteps=1000)
    act.save('./move_to_beacon')

def main(argv):
    env = make('MoveToBeacon-bbueno5000-v0')
    network = 'mlp'
    train_deepq_agent(env, network)

if __name__ == '__main__':
    app.run(main)
