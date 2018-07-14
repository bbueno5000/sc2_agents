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

import collections
import tensorflow as tf


class _Function:

    def __init__(self, inputs, outputs, updates, givens):
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (type(inpt) is tf.Tensor and len(inpt.op.inputs) == 0):
                assert False, "Inputs should all be placeholders, constants, or have a make_feed_dict method."
        self.inputs = inputs
        self.updates = updates or []
        self.update_group = tf.group(*self.updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def __call__(self, *args):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input_(feed_dict, inpt, value)
        # Update feed dict with givens
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        return tf.get_default_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]

    def _feed_input_(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value


def function(inputs, outputs, updates=None, givens=None):
    """
    Just like Theano function.

    Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs.
    Function f takes values to be fed to the input's placeholders and
    produces the values of the expressions in outputs.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()
            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    inputs: _[tf.placeholder, tf.constant, or object with make_feed_dict method]_
        list of input arguments
    outputs: _[tf.Variable] or tf.Variable_
        list of outputs or a single output to be returned from function.
        Returned value will also have the same shape.
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


def huber_loss(x, delta=1.0):
    """
    (https://en.wikipedia.org/wiki/Huber_loss)
    """
    return tf.where(tf.abs(x) < delta, tf.square(x) * 0.5, delta * (tf.abs(x) - 0.5 * delta))
