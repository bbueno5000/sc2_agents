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
This file contains a DeepQ training algorithm
derived from openai baselines for training SC2 agents.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from baselines.deepq.replay_buffer import ReplayBuffer
from numpy import abs as np_abs
from numpy import log as np_log    # pylint: disable=E0611
from numpy import ones_like as np_ones_like
from os import path
from starcraft_agents.deepq import act_wrapper
from starcraft_agents.deepq import mlp_models as deepq_models
from starcraft_agents.deepq.deepq_graph import ConstructDeepQGraph
from tempfile import TemporaryDirectory
from tensorflow import get_default_session
from tensorflow import global_variables_initializer
from tensorflow import Session
from tensorflow import summary as tf_summary
from tensorflow import train as tf_train


class Learn:
    """
    Train a deepq model.

    batch_size: _int_
        Size of a batched sampled from
        replay buffer for training.
    buffer_size: _int_
        Size of the replay buffer.
    checkpoint_freq: _int_
        How often to save the model.
    exploration_final_eps: _float_
        Final value of random action probability.
    exploration_fraction: _float_
        Fraction of entire training period
        over which the exploration rate is annealed.
    env: _pysc2.env.SC2Env_
        Environment to train in.
    learning_starts: _int_
        How many steps of the model to collect
        transitions before learning starts
    max_timesteps: _int_
        Number of environment steps to optimize.
    num_episodes: _int_
        Number of episodes to run.
    print_freq: _int_
        The episodic frequency in which to print an update.
        None to disable.
    prioritized_replay: _bool_
        If True prioritized replay buffer will be used.
    prioritized_replay_alpha: _float_
        Alpha value for prioritized replay buffer.
    prioritized_replay_beta: _float_
        Initial value of beta for prioritized replay buffer.
    prioritized_replay_beta_iters: _int_
        Number of iterations over which beta
        will be annealed from initial value to 1.0.
        If set to None equals max_timesteps.
    prioritized_replay_eps: _float_
        Epsilon value to add to the TD errors
        when updating priorities.
    q_func: _tf.Variable, int, str, bool_
        The model that takes the inputs.
    target_network_update_freq: _int_
        Update the target network.
    train_freq: _int_
        Update the model.
        None to disable.
    """
    def __init__(self,
                 agent_cls,
                 cnn_to_mlp_args,
                 env,
                 map_name,
                 num_episodes,
                 save_replay,
                 learning_rate=0.0005,
                 num_actions=64):
        model = deepq_models.cnn_to_mlp(*cnn_to_mlp_args)
        graph_kwargs = {'double_q': True,
                        'optimizer': tf_train.AdamOptimizer(learning_rate),
                        'q_func': model}
        construct_graph = ConstructDeepQGraph(num_actions)
        act_x, self.train_x, self.update_target_x, _ = construct_graph(scope="deepq_x", **graph_kwargs)
        act_y, self.train_y, self.update_target_y, _ = construct_graph(scope="deepq_y", **graph_kwargs)
        act_params_x = {'num_actions': num_actions,
                        'q_func': model,
                        'reuse': False,
                        'scope': "deepq_x"}
        act_params_y = {'num_actions': num_actions,
                        'q_func': model,
                        'reuse': False,
                        'scope': "deepq_y"}
        act_x = act_wrapper.ActWrapper(act_x, act_params_x)
        act_y = act_wrapper.ActWrapper(act_y, act_params_y)
        self.agent = agent_cls(act_x, act_y)
        self.agent_name = agent_cls.__name__
        self.agent.reset()
        self.agent.setup(env.action_spec(), env.observation_spec())
        self.construct_graph = construct_graph
        self.env = env
        self.map_name = map_name
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.reset = True
        self.saved_mean_reward = 0
        self.save_replay = save_replay

    def __call__(self,
                 act_ops_dir,
                 checkpoints_dir,
                 tensorboard_dir,
                 batch_size=32,
                 buffer_size=50000,
                 episode_minimum=2,
                 exploration_final_eps=0.02,
                 exploration_fraction=0.1,
                 learning_starts=250,
                 max_timesteps=1000,
                 param_noise=False,
                 print_freq=1,
                 prioritized_replay=False,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta=0.4,
                 prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6,
                 target_network_update_freq=250,
                 train_freq=10):
        sess = Session()
        sess.run(global_variables_initializer())
        sess.__enter__()
        train_writer_op = tf_summary.FileWriter(tensorboard_dir, sess.graph)
        timesteps = self.env.reset()
        screen = self.agent.screen(timesteps[0].observation)
        self.update_target_x()
        self.update_target_y()
        if prioritized_replay:
            replay_buffer_x = PrioritizedReplayBuffer(prioritized_replay_alpha, buffer_size)
            replay_buffer_y = PrioritizedReplayBuffer(prioritized_replay_alpha, buffer_size)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = max_timesteps
            beta_schedule = LinearSchedule(prioritized_replay_beta_iters, 1.0,  prioritized_replay_beta)
        else:
            replay_buffer_x = ReplayBuffer(buffer_size)
            replay_buffer_y = ReplayBuffer(buffer_size)
            beta_schedule = None
        # create the schedule for exploration starting from 1 (every action is random)
        # down to 0.5 (50% of actions are selected according to values predicted by the model)
        exploration = LinearSchedule(int(exploration_fraction * max_timesteps), exploration_final_eps)
        # begin training process
        with TemporaryDirectory() as temp_dir:
            # TRAINING LOOP
            for global_step in range(max_timesteps):
                # graduation requirements
                if self._callback is not None:
                    if self._callback():
                        break
                step_kwargs = {}
                if not param_noise:
                    update_eps = exploration.value(global_step)
                    update_param_noise_threshold = 0.0
                else:
                    update_eps = 0.0
                    # compute the threshold such that the KL divergence between perturbed
                    # and non-perturbed policy is comparable to eps-greedy exploration
                    # with eps = exploration.value(global_step)
                    # See Appendix C.1 in Parameter Space Noise for Exploration,
                    # Plappert et al., 2017 for detailed explanation.
                    update_param_noise_threshold = (-np_log(1.0 -
                                                    exploration.value(global_step) +
                                                    exploration.value(global_step) /
                                                    float(self.num_actions)))
                    step_kwargs['reset'] = self.reset
                    step_kwargs['update_param_noise_scale'] = True
                    step_kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                step_kwargs['update_eps'] = update_eps
                training_step = self.agent.training_step(timesteps[0], **step_kwargs)
                timesteps = self.env.step([training_step])
                new_screen = self.agent.screen(timesteps[0].observation)
                episode_completed = timesteps[0].last()
                self.reset = False
                # REPLAY BUFFERS
                replay_buffer_x.add(action=self.agent.x_coord,
                                    done=float(episode_completed),
                                    obs_t=screen,
                                    obs_tp1=new_screen,
                                    reward=timesteps[0].reward)
                replay_buffer_y.add(action=self.agent.y_coord,
                                    done=float(episode_completed),
                                    obs_t=screen,
                                    obs_tp1=new_screen,
                                    reward=timesteps[0].reward)
                screen = new_screen
                # END OF EPISODE
                if episode_completed:
                    self._save_model_checkpoint(path.join(temp_dir, checkpoints_dir) + "_ep_" + str(self.agent.episodes))
                    timesteps, screen = self._reset()
                # Q-LEARNING
                # minimize the error in Bellman's equation on a batch sampled from replay buffer
                if global_step > learning_starts and global_step % train_freq == 0:
                    if prioritized_replay:
                        experience = replay_buffer_x.sample(batch_size, beta_schedule.value(global_step))
                        (obs_t_x, actions_x, rewards_x, obs_tp1_x, dones_x, weights_x, batch_idxes_x) = experience
                        experience = replay_buffer_y.sample(batch_size, beta_schedule.value(global_step))
                        (obs_t_y, actions_y, rewards_y, obs_tp1_y, dones_y, weights_y, batch_idxes_y) = experience
                    else:
                        obs_t_x, actions_x, rewards_x, obs_tp1_x, dones_x = replay_buffer_x.sample(batch_size)
                        weights_x, batch_idxes_x = np_ones_like(rewards_x), None
                        obs_t_y, actions_y, rewards_y, obs_tp1_y, dones_y = replay_buffer_y.sample(batch_size)
                        weights_y, batch_idxes_y = np_ones_like(rewards_y), None
                    td_errors_x = self.train_x(obs_t_x, actions_x, rewards_x, obs_tp1_x, dones_x, weights_x)
                    td_errors_y = self.train_y(obs_t_y, actions_y, rewards_y, obs_tp1_y, dones_y, weights_y)
                    # PRIORITIZED REPLAY
                    if prioritized_replay:
                        new_priorities = np_abs(td_errors_x) + prioritized_replay_eps
                        replay_buffer_x.update_priorities(batch_idxes_x, new_priorities)
                        new_priorities = np_abs(td_errors_y) + prioritized_replay_eps
                        replay_buffer_y.update_priorities(batch_idxes_y, new_priorities)
                if global_step > learning_starts:
                    if global_step % target_network_update_freq == 0:
                        self.update_target_x()
                        self.update_target_y()
                self._log_update(episode_completed, exploration, global_step, print_freq)
                self._save_mean_reward(episode_minimum, global_step, learning_starts)
        train_writer_op.close()
        self._save_act_ops(act_ops_dir)
        if self.save_replay:
            self.env.save_replay(self.agent_name)

    def _callback(self):
        return self.agent.episodes == self.num_episodes

    def _log_update(self, episode_completed, exploration, global_step, print_freq):
        self.agent.mean_reward = round(self.agent.reward / self.agent.episodes, 1)
        if episode_completed and print_freq is not None and self.agent.episodes % print_freq == 0:
            logger.record_tabular("% time spent exploring ", int(100 * exploration.value(global_step)))
            logger.record_tabular("mean reward", self.agent.mean_reward)
            logger.record_tabular("steps", global_step)
            logger.dump_tabular()

    def _reset(self):
        self.reset = True
        self.agent.reset()
        timesteps = self.env.reset()
        screen = self.agent.screen(timesteps[0].observation)
        return timesteps, screen

    def _save_act_ops(self, dir_name):
        logger.log("Saving act ops with mean reward: {} ".format(self.agent.mean_reward))
        self.agent.act_x.save_model(path.join(dir_name, self.map_name + "_x.pkl"))
        self.agent.act_y.save_model(path.join(dir_name, self.map_name + "_y.pkl"))

    def _save_mean_reward(self, episode_minimum, global_step, learning_starts):
        if self.agent.episodes > episode_minimum and global_step > learning_starts:
            if self.agent.mean_reward > self.saved_mean_reward:
                msg = "Saving reward due to mean reward increase: {} -> {}"
                logger.log(msg.format(self.saved_mean_reward, self.agent.mean_reward))
                self.saved_mean_reward = self.agent.mean_reward

    def _save_model_checkpoint(self, file_path):
        logger.log("Saving model checkpoint.")
        saver = tf_train.Saver()
        saver.save(get_default_session(), file_path)
