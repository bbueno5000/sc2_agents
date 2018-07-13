"""
Deep Q learning graph.

The functions in this file are used to create the following:

======= act ========

    Function to chose an action given an observation.

======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    __Parameters__
    action: _np.array_
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: _np.array_
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_t: _object_
        a batch of observations
    obs_tp1: _object_
        observations that followed obs_t
    done: _np.array_
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: _np.array_
        imporance weights for every element of the batch
        (gradient is multiplied by the importance weight)
        dtype must be float32 and shape must be (batch_size,)

    __Returns__
    td_error: _np.array_ a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    Copy the parameters from optimized Q function to the target Q function.
    In Q learning we optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning.
    For example, for Atari Q' is set to Q once every 10000 updates training steps.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pysc2.deepq.utils import function as utils_function
from pysc2.deepq.utils import huber_loss
from pysc2.deepq.act_function import ConstructActFunc
from pysc2.deepq.act_function import ConstructActFuncWithNoise
from tensorflow import argmax as tf_argmax
from tensorflow import clip_by_norm as tf_clip_by_norm
from tensorflow import float32 as tf_float32
from tensorflow import get_collection
from tensorflow import get_variable_scope
from tensorflow import GraphKeys
from tensorflow import group as tf_group
from tensorflow import int32 as tf_int32
from tensorflow import one_hot as tf_one_hot
from tensorflow import placeholder as tf_placeholder
from tensorflow import reduce_max as tf_reduce_max
from tensorflow import reduce_mean as tf_reduce_mean
from tensorflow import reduce_sum as tf_reduce_sum
from tensorflow import stop_gradient as tf_stop_gradient
from tensorflow import Variable as tf_Variable
from tensorflow import variable_scope as tf_variable_scope


class ConstructDeepQGraph:
    """
    double_q: _bool_
        if true will use Double Q Learning.
        (https://arxiv.org/abs/1509.06461)
        In general it is a good idea to keep it enabled.
    gamma: _float_
        discount rate.
    grad_norm_clipping: _float or None_
        clip gradient norms to this value.
        If None no clipping is performed.
    q_func: _(tf_Variable, int, str, bool) -> tf_Variable_
        the model that takes the following inputs:
            observation_in: _object_
                the output of observation placeholder
            num_actions: _int_
                number of actions
            scope: _str_
            reuse: _bool_
                should be passed to outer variable scope
                and returns a tensor of shape
                (batch_size, num_actions)
                with values of every action.
    num_actions: _int_
        number of actions
    optimizer: _tf.train.Optimizer_
        optimizer to use for the Q-learning objective.
    param_noise: _bool_
        whether or not to use parameter space noise
        (https://arxiv.org/abs/1706.01905)
    param_noise_filter_func: _tf.Variable -> bool_
        function that decides whether or
        not a variable should be perturbed.
        Only applicable if param_noise is True.
        If set to None, default_param_noise_filter is used.
    reuse: _bool or None_
        whether or not the variables should be reused.
        To be able to reuse the scope must be given.
    scope: _str or VariableScope_
        optional scope for variable_scope.
    act: _(tf_Variable, bool, float) -> tf_Variable_
        function to select and action given observation.
        See the top of the file for details.
    debug: _{str: function}_
        a bunch of functions to print debug data like q_values.
    train: _(object, np.array, np.array, object, np.array, np.array) -> np.array_
        optimize the error in Bellman's equation.
        See the top of the file for details.
    update_target: _() -> ()_
        copy the parameters from optimized Q
        function to the target Q function.
        See the top of the file for details.
    """
    def __init__(self, num_actions):
        self.act_t_ph = tf_placeholder(tf_int32, [None], "action")
        self.done_mask_ph = tf_placeholder(tf_float32, [None], "done")
        self.importance_weights_ph = tf_placeholder(tf_float32, [None], "weight")
        self.num_actions = num_actions
        self.obs_t_input = tf_placeholder(tf_float32, [None] + list((64, 64)), name="obs_t")
        self.obs_tp1_input = tf_placeholder(tf_float32, [None] + list((64, 64)), name="obs_tp1")
        self.rew_t_ph = tf_placeholder(tf_float32, [None], "reward")

    def __call__(self,
                 double_q,
                 optimizer,
                 q_func,
                 scope,
                 gamma=1.0,
                 grad_norm_clipping=None,
                 param_noise=False,
                 param_noise_filter_func=None,
                 reuse=False):
        # ACT FUNCTION
        if param_noise:
            act_func = ConstructActFuncWithNoise(self.num_actions)(q_func, reuse, scope)
        else:
            act_func = ConstructActFunc(self.num_actions)(q_func, reuse, scope)
        with tf_variable_scope(scope, reuse=reuse):
            # TRAIN FUNCTION
            q_func_vars = get_collection(GraphKeys.GLOBAL_VARIABLES, get_variable_scope().name + "/q_func")
            # q scores for actions which we know were selected in the given state.
            q_t = q_func(self.obs_t_input, self.num_actions, scope="q_func", reuse=True)    # reuse parameters from act
            inputs = self._inputs()
            outputs = self._outputs(double_q, gamma, q_func, q_t)
            self.updates = self._updates(grad_norm_clipping, optimizer, outputs, q_func_vars)
            train_func = utils_function(inputs, outputs, [self.updates])
            # UPDATE TARGET FUNCTION
            # update_target function will be called periodically to copy Q network to target Q network
            update_target_expr = []
            target_q_func_vars = get_collection(GraphKeys.GLOBAL_VARIABLES, get_variable_scope().name + "/target_q_func")
            for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_expr.append(var_target.assign(var))
            update_target_expr = tf_group(*update_target_expr)
            update_target_func = utils_function([], [], [update_target_expr])
            # Q VALUES FUNCTION
            q_values_func = utils_function([self.obs_t_input], q_t)
            return act_func, train_func, update_target_func, {'q_values': q_values_func}

    def _inputs(self):
        return [self.obs_t_input,
                self.act_t_ph,
                self.rew_t_ph,
                self.obs_tp1_input,
                self.done_mask_ph,
                self.importance_weights_ph]

    def _outputs(self, double_q, gamma, q_func, q_t):
        # q network evaluatios
        q_t_selected = tf_reduce_sum(q_t * tf_one_hot(self.act_t_ph, self.num_actions), 1)
        # compute estimate of best possible value starting from state at t + 1
        q_tp1 = q_func(self.obs_tp1_input, self.num_actions, scope="target_q_func")
        if double_q:
            q_tp1_using_online_net = q_func(self.obs_tp1_input, self.num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf_argmax(q_tp1_using_online_net, 1)
            q_tp1_best = tf_reduce_sum(q_tp1 * tf_one_hot(q_tp1_best_using_online_net, self.num_actions), 1)
        else:
            q_tp1_best = tf_reduce_max(q_tp1, 1)
        # target q network evalution
        q_tp1_best_masked = (1.0 - self.done_mask_ph) * q_tp1_best
        # compute RHS of bellman equation
        q_t_selected_target = self.rew_t_ph + gamma * q_tp1_best_masked
        # compute the td_error (potentially clipped)
        return q_t_selected - tf_stop_gradient(q_t_selected_target)

    def _updates(self, grad_norm_clipping, optimizer, td_error, q_func_vars):
        # compute optimization operation (potentially with gradient clipping)
        errors = huber_loss(td_error)
        weighted_error = tf_reduce_mean(self.importance_weights_ph * errors)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf_clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)
        return optimize_expr
