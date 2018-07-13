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
Wrapper for saving and loading act functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dill import dump
from dill import load
from os import makedirs
from os import path
from os import walk
from pysc2.deepq.act_function import ConstructActFunc
from tempfile import TemporaryDirectory
from tensorflow import get_default_session
from tensorflow import Session
from tensorflow import train as tf_train
from zipfile import ZIP_DEFLATED
from zipfile import ZipFile


class ActWrapper:

    def __init__(self, act, act_params):
        self.act = act
        self.act_params = act_params

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    @staticmethod
    def load_model(path_name):
        """
        path_name:
            Path to saved model.
        return:
            Wrapper over act function.
            Adds ability to save and load.
        """
        with open(path_name, "rb") as f:
            model_data, act_params = load(f)
        args = act_params.pop('num_actions', "Key not found.")
        act = ConstructActFunc(*args)(**act_params)
        sess = Session()
        sess.__enter__()
        with TemporaryDirectory() as temp_dir:
            arc_path = path.join(temp_dir, "packed.zip")
            with open(arc_path, "wb") as file:
                file.write(model_data)
            ZipFile(arc_path, 'r', ZIP_DEFLATED).extractall(temp_dir)
            fname = path.join(temp_dir, "model")
            saver = tf_train.Saver()
            saver.restore(get_default_session(), fname)
        return ActWrapper(act, act_params)

    def save_model(self, path_name):
        """
        Save model to a pickle located at `path_name`.
        """
        with TemporaryDirectory() as temp_dir:
            arc_path = path.join(temp_dir, "packed.zip")
            with ZipFile(arc_path, 'w') as zipf:
                for root, _, files in walk(temp_dir):
                    for fname in files:
                        file_path = path.join(root, fname)
                        if file_path != arc_path:
                            zipf.write(file_path, path.relpath(file_path, temp_dir))
            with open(arc_path, 'rb') as f:
                model_data = f.read()
        makedirs(path.dirname(path_name), exist_ok=True)
        with open(path_name, 'wb') as f:
            dump((model_data, self.act_params), f)


def load_model(path_name):
    """
    Load act function that was returned by learn function.

    path:
        Path to the pickle.
    return:
        Function that takes a batch of observations and returns actions.
    """
    return ActWrapper.load_model(path_name)
