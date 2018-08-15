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
from gym import make
from gym_sc2 import envs

FLAGS = flags.FLAGS
flags.DEFINE_string('algorithm', None, "Id of the Gym environment")
flags.DEFINE_string('gym_id', None, "Id of the Gym environment")
flags.DEFINE_string('network', None, "Agent configuration file")
flags.DEFINE_string('save_dir', None, "Id of the Gym environment")

def train_deepq_agent(env, network):
    from baselines import deepq
    act = deepq.learn(
        env,
        network,
        total_timesteps=1000)
    act.save(FLAGS.save_dir)

def train_ppo_agent(env, network):
    from baselines import ppo1
    def policy_fn(name, ob_space, ac_space):    # pylint: disable=W0613
        return ppo1.cnn_policy.CnnPolicy(name, ob_space, ac_space)
    ppo1.pposgd_simple.learn(
        env,
        policy_fn,
        max_timesteps=1000,
        timesteps_per_actorbatch=2048,
        clip_param=0.2,
        entcoeff=0.0,
        optim_epochs=5,
        optim_stepsize=3e-4,
        optim_batchsize=256,
        gamma=0.99,
        lam=0.95,
        schedule='linear')

def main(argv):
    env = make(FLAGS.gym_id)
    if FLAGS.algorithm is 'deepq':
        train_deepq_agent(env, FLAGS.network)
    else:
        train_ppo_agent(env, FLAGS.network)
    env.close()

if __name__ == '__main__':
    FLAGS.algorithm = 'deepq'
    FLAGS.gym_id = 'MoveToBeacon-bbueno5000-v0'
    FLAGS.network = 'cnn'
    FLAGS.save_dir = './move_to_beacon-deepq-cnn-1'
    app.run(main)
