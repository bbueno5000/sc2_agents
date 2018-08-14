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

from absl import flags
from argparse import ArgumentParser
from move_to_beacon_1d import MoveToBeacon1d

FLAGS = flags.FLAGS
FLAGS([__file__])

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--visualize',
        type=bool,
        default=False,
        help='show the pysc2 visualizer')
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='number of episodes to run')
    parser.add_argument(
        '--step-mul',
        type=int,
        default=None,
        help='number of game steps to take per turn')
    args = parser.parse_args()
    example = MoveToBeacon1d(args.visualize, args.step_mul)
    rewards = example.run(args.num_episodes)
    print('Total reward: {}'.format(rewards.sum()))
    print('Average reward: {} +/- {}'.format(rewards.mean(), rewards.std()))
    print('Minimum reward: {}'.format(rewards.min()))
    print('Maximum reward: {}'.format(rewards.max()))

if __name__ == "__main__":
    main()
