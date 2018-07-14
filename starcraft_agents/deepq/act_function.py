"""
Deep Q learning graph.

The functions in this file are used to create the following:

======= act ========

    Function to chose an action given an observation.
    observation: Observation that can be fed into the output of make_obs_ph
    stochastic: if set to False all the actions are always deterministic
        (default False)
    update_eps_ph: update epsilon a new value, if negative not update happens
        (default: no update)
    return: Tensor of dtype tf.int64 and shape (BATCH_SIZE,)
        with an action to be performed for every element of the batch.

======= act with parameter noise ========

    Creates the act function with support for
    parameter space noise exploration.
    Function to chose an action given an observation.
    (https://arxiv.org/abs/1706.01905)
    observation: Observation that can be fed into the output of make_obs_ph
    stochastic: if set to False all the actions are always deterministic
        (default False)
    update_eps_ph: update epsilon a new value, if negative not update happens
        (default: no update)
    reset_ph: reset the perturbed policy by sampling a new perturbation
    update_param_noise_threshold_ph: the desired threshold for the difference
        between non-perturbed and perturbed policy
    update_param_noise_scale_ph: whether or not to update the scale of
        the noise for the next time it is re-perturbed
    return: Tensor of dtype tf.int64 and shape (BATCH_SIZE,)
        with an action to be performed for every element of the batch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from starcraft_agents.deepq.utils import function as utils_function
import tensorflow as tf


class ConstructActFunc:
    """
    observation: Observation that can be fed into the output of make_obs_ph
    stochastic: if set to False all the actions are always deterministic
        (default False)
    update_eps_ph: update epsilon a new value, if negative no update happens
        (default: no update)
    num_actions: number of actions
    q_func: _(tf.Variable, int, str, bool) -> tf.Variable_
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
    reuse: whether or not the variables should be reused.
        To be able to reuse the scope must be given.
    scope: optional scope for variable_scope.
    return: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
        Tensor of dtype tf.int64 and shape (BATCH_SIZE,)
        with an action to be performed for
        every element of the batch.
    """

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.observations_ph = tf.placeholder(tf.float32, [None] + list((64, 64)), name="observation")
        self.stochastic_ph = tf.placeholder(tf.bool, (), "stochastic")
        self.update_eps_ph = tf.placeholder(tf.float32, (), "update_eps")

    def __call__(self, q_func, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            # inputs
            inputs = [self.observations_ph, self.stochastic_ph, self.update_eps_ph]
            # outputs
            batch_size = tf.shape(self.observations_ph)[0]
            eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
            choose_random = tf.random_uniform(tf.stack([batch_size]), 0, 1, tf.float32) < eps
            random_actions = tf.random_uniform(tf.stack([batch_size]), 0, self.num_actions, tf.int64)
            q_values = q_func(self.observations_ph, self.num_actions, scope="q_func")
            deterministic_actions = tf.argmax(q_values, axis=1)
            stochastic_actions = tf.where(choose_random, random_actions, deterministic_actions)
            outputs = tf.cond(self.stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
            # updates
            update_eps_expr = eps.assign(tf.cond(self.update_eps_ph >= 0,
                                                 lambda: self.update_eps_ph,
                                                 lambda: eps))
            updates = [update_eps_expr]
            # givens
            givens = {self.stochastic_ph: True, self.update_eps_ph: -1.0}
            # functions
            self.act = utils_function(inputs, outputs, updates, givens)
            return self._act

    def _act(self, observation, stochastic=True, update_eps=-1):
        return self.act(observation, stochastic, update_eps)


class ConstructActFuncWithNoise:
    """
    num_actions: number of actions
    observation: Observation that can be feed into the output of make_obs_ph
    original_scope: _str_
    param_noise_filter_func: _tf.Variable -> bool_
        function that decides whether or not a
        variable should be perturbed.
        Only applicable if param_noise is True.
        If set to None, default_param_noise_filter
        is used by default.
    param_noise_scale: _tf variable_
    param_noise_threshold: _tf variable_
    perturbed_scope: _str_
    q_func: _(tf.Variable, int, str, bool) -> tf.Variable_
            the model that takes the following inputs:
                observation_in: _object_
                    the output of observation placeholder
                num_actions: _int_
                    number of actions
                reuse: _bool_
                    should be passed to outer variable scope
                    and returns a tensor of shape
                    (batch_size, num_actions)
                    with values of every action.
                scope: _str_
    reuse: _bool or None_
        whether or not the variables should be reused.
        To be able to reuse the scope must be given.
    reset_ph: _bool_
        reset the perturbed policy by
        sampling a new perturbation
    scope: _str or VariableScope_
        optional scope for variable_scope.
    stochastic: _bool_
        if set to False all the actions
        are always deterministic (default False)
    update_eps_ph: _float_
        update epsilon a new value,
        if negative not update happens
        (default: no update)
    update_param_noise_threshold_ph: _float_
        the desired threshold for the difference
        between non-perturbed and perturbed policy
    update_param_noise_scale_ph: _bool_
        whether or not to update the scale of the
        noise for the next time it is re-perturbed
    return: _(tf.Variable, bool, float, bool, float, bool) -> tf.Variable_
        function to select and action given observation.
        Tensor of dtype tf.int64 and shape (BATCH_SIZE,)
        with an action to be performed for every element of the batch.
    """
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.observations_ph = tf.placeholder(tf.float32, [None] + list((64, 64)), name="observation")
        self.reset_ph = tf.placeholder(tf.bool, (), "reset")
        self.stochastic_ph = tf.placeholder(tf.bool, (), "stochastic")
        self.update_eps_ph = tf.placeholder(tf.float32, (), "update_eps")
        self.update_param_noise_scale_ph = tf.placeholder(tf.bool, (), "update_param_noise_scale")
        self.update_param_noise_threshold_ph = tf.placeholder(tf.float32, (), "update_param_noise_threshold")

    def __call__(self, q_func, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            # inputs
            inputs = [self.observations_ph,
                      self.stochastic_ph,
                      self.update_eps_ph,
                      self.reset_ph,
                      self.update_param_noise_threshold_ph,
                      self.update_param_noise_scale_ph]
            # outputs
            batch_size = tf.shape(self.observations_ph.get())[0]
            eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
            chose_random = tf.random_uniform(tf.stack([batch_size]), 0, 1) < eps
            random_actions = tf.random_uniform(tf.stack([batch_size]), 0, self.num_actions, tf.int64)
            # Perturbable Q used for the actual rollout.
            q_values_perturbed = q_func(self.observations_ph.get(), self.num_actions, scope="perturbed_q_func")
            deterministic_actions = tf.argmax(q_values_perturbed, axis=1)
            stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
            outputs = tf.cond(self.stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
            # updates
            update_eps_expr = eps.assign(tf.cond(self.update_eps_ph >= 0, lambda: self.update_eps_ph, lambda: eps))
            # Functionality to update the threshold for parameter space noise.
            param_noise_threshold = tf.get_variable(initializer=tf.constant_initializer(0.05),
                                                    name="param_noise_threshold",
                                                    shape=(),
                                                    trainable=False)
            update_param_noise_threshold_expr = param_noise_threshold.assign(tf.cond(self.update_param_noise_threshold_ph >= 0,
                                                                             lambda: self.update_param_noise_threshold_ph,
                                                                             lambda: param_noise_threshold))
            updates = [update_eps_expr,
                       tf.cond(self.reset_ph, lambda: self.perturb_vars(), lambda: tf.group(*[])),
                       tf.cond(self.update_param_noise_scale_ph,
                               lambda: self.update_scale(param_noise_threshold, q_func),
                               lambda: tf.Variable(0.0, False)),
                       update_param_noise_threshold_expr]
            # givens
            givens = {self.reset_ph: False,
                      self.stochastic_ph: True,
                      self.update_eps_ph: -1.0,
                      self.update_param_noise_scale_ph: False,
                      self.update_param_noise_threshold_ph: False}
            # functions
            self.act_function = utils_function(inputs, outputs, updates, givens)
            return self.act

    def absolute_scope_name(self, relative_scope_name):
        """
        Appends parent scope name to `relative_scope_name`
        """
        return self.scope_name() + "/" + relative_scope_name

    def act(self, obs, stochastic=True, update_eps=-1):
        return self.act_function(obs, stochastic, update_eps)

    def default_param_noise_filter(self, var):
        # do not perturb untrainable variables
        if var not in tf.trainable_variables():
            return False
        # perturb fully-connected layers
        if "fully_connected" in var.name:
            return True
        # The remaining layers are likely convolutional or layer norm layers,
        # which we do not wish to perturb.
        # In the former case because they only extract features.
        # In the latter case because we use them for normalization purposes.
        # If you change your network, you will likely want
        # to consider which layers to perturb and which to keep untouched.
        return False

    def perturb_vars(self,
                     param_noise_filter_func=None,
                     param_noise_scale=None,
                     original_scope="q_func",
                     perturbed_scope="adaptive_q_func"):
        """
        We have to wrap this code into a function due to the way tf.cond() works.
        (https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond)
        """
        all_vars = self.scope_vars(self.absolute_scope_name(original_scope))
        all_perturbed_vars = self.scope_vars(self.absolute_scope_name(perturbed_scope))
        assert len(all_vars) == len(all_perturbed_vars)
        perturb_ops = []
        for var, perturbed_var in zip(all_vars, all_perturbed_vars):
            if param_noise_filter_func(perturbed_var):    # Perturb this variable.
                op = tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), 0.0, param_noise_scale))
            else:    # Do not perturb, just assign.
                op = tf.assign(perturbed_var, var)
            perturb_ops.append(op)
        assert len(all_vars) == len(perturb_ops)
        return tf.group(*perturb_ops)

    def scope_name(self):
        return tf.get_variable_scope().name

    def scope_vars(self, scope='', trainable_only=False):
        """
        Get variables inside a scope.
        The scope can be specified as a string.
        scope: scope in which the variables reside.
        trainable_only: whether or not to return only the variables that were marked as trainable.
        return: _[tf.Variable]_ list of variables in `scope`.
        """
        key = tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES
        scope = scope if isinstance(scope, str) else scope.name
        return tf.get_collection(key, scope)

    def update_scale(self, param_noise_threshold, q_func, param_noise_filter_func=None):
        # Unmodified Q.
        q_values = q_func(self.observations_ph.get(), self.num_actions, scope="q_func")
        var1 = tf.nn.softmax(q_values)
        var2 = tf.log(tf.nn.softmax(q_values))
        # Set up functionality to re-compute `param_noise_scale`.
        # This perturbs yet another copy of the network and measures
        # the effect of that perturbation in action space.
        # If the perturbation is too big, reduce scale of perturbation, otherwise increase.
        q_values_adaptive = q_func(self.observations_ph.get(), self.num_actions, scope="adaptive_q_func")
        var3 = tf.log(tf.nn.softmax(q_values_adaptive))
        mean_kl = tf.reduce_mean(tf.reduce_sum(input_tensor=var1 * (var2 - var3), axis=-1))
        if param_noise_filter_func is None:
            param_noise_filter_func = self.default_param_noise_filter
        param_noise_scale = tf.get_variable(initializer=tf.constant_initializer(0.01),
                                            name="param_noise_scale",
                                            shape=(),
                                            trainable=False)
        perturb_for_adaption = self.perturb_vars(param_noise_filter_func, param_noise_scale)
        with tf.control_dependencies([perturb_for_adaption]):
            update_scale_expr = tf.cond(mean_kl < param_noise_threshold,
                                       lambda: param_noise_scale.assign(param_noise_scale * 1.01),
                                       lambda: param_noise_scale.assign(param_noise_scale / 1.01))
        return update_scale_expr
