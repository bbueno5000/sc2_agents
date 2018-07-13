from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow import expand_dims
from tensorflow import nn as tf_nn
from tensorflow import reduce_mean as tf_reduce_mean
from tensorflow import variable_scope
from tensorflow.contrib import layers    # pylint: disable=E0611


def _cnn_to_mlp(convs, dueling, hiddens, inpt, layer_norm, num_actions, reuse, scope):
    with variable_scope(scope, reuse=reuse):
        out = inpt
        with variable_scope("convnet"):
            for kernel_size, num_outputs, stride in convs:
                out = layers.convolution2d(activation_fn=tf_nn.relu,
                                           inputs=out,
                                           kernel_size=kernel_size,
                                           num_outputs=num_outputs,
                                           stride=stride)
        conv_out = layers.flatten(out)
        with variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf_nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)
        if dueling:
            with variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf_nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf_reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def _mlp(hiddens, inpt, layer_norm, num_actions, reuse, scope):
    with variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf_nn.relu(out)
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return q_out


def cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False):
    """
    This model takes input an observation and returns values of all actions.
    convs: _[(int, int int)]_
         list of convolutional layers
         (kernel_size, num_outputs, stride)
    dueling: _bool_
        if true double the output MLP
        to compute a baseline for action scores.
    hiddens: _[int]_
        list of sizes of hidden layers
    layer_norm: _bool_
    q_func: q_function for DQN algorithm.
    """
    return lambda inpt, num_actions, scope, reuse=False: _cnn_to_mlp(convs,
                                                                      dueling,
                                                                      hiddens,
                                                                      inpt,
                                                                      layer_norm,
                                                                      num_actions,
                                                                      reuse,
                                                                      scope)


def mlp(hiddens, layer_norm=False):
    """
    This model takes as input an observation and returns values of all actions.
    hiddens: _[int]_
        list of sizes of hidden layers
    q_func: q_function for DQN algorithm.
    """
    return lambda inpt, num_actions, scope, reuse=False: _mlp(hiddens,
                                                               inpt,
                                                               layer_norm,
                                                               num_actions,
                                                               reuse,
                                                               scope)
