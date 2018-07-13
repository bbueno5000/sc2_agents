from collections import OrderedDict as collections_OrderedDict
from tensorflow import abs as tf_abs
from tensorflow import cast as tf_cast
from tensorflow import float32 as tf_float32
from tensorflow import get_default_session as tf_get_default_session
from tensorflow import group as tf_group
from tensorflow import placeholder as tf_placeholder
from tensorflow import square as tf_square
from tensorflow import Tensor as tf_Tensor
from tensorflow import uint8 as tf_uint8
from tensorflow import where as tf_where


class _Function:

    def __init__(self, inputs, outputs, updates, givens):
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (type(inpt) is tf_Tensor and len(inpt.op.inputs) == 0):
                assert False, "Inputs should all be placeholders, constants, or have a make_feed_dict method."
        self.inputs = inputs
        self.updates = updates or []
        self.update_group = tf_group(*self.updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def __call__(self, *args):
        assert len(args) <= len(self.inputs), "Too many arguments provided."
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input_(feed_dict, inpt, value)
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        return tf_get_default_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]

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
        x = tf_placeholder(tf_int32, (), name="x")
        y = tf_placeholder(tf_int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()
            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    inputs: _[tf_placeholder, tf_constant, or object with make_feed_dict method]_
        list of input arguments
    outputs: _[tf_Variable] or tf_Variable_
        list of outputs or a single output to be returned from function.
        Returned value will also have the same shape.
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections_OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


def huber_loss(x, delta=1.0):
    """
    (https://en.wikipedia.org/wiki/Huber_loss)
    """
    return tf_where(tf_abs(x) < delta, tf_square(x) * 0.5, delta * (tf_abs(x) - 0.5 * delta))
