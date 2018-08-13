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
Train DQN agent.
"""

from absl import app
from absl import flags
from dill import dump as pkl_dump
from dill import load as pkl_load
from future.builtins import range as builtins_range
from importlib import import_module
from json import dumps as json_dumps
from json import loads as json_loads
from mpyq import MPQArchive as mpyq_MPQArchive
from os import makedirs
from os import path
from platform import system as platform_system
from pysc2 import maps
from pysc2 import run_configs
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.lib import renderer_human
from s2clientprotocol import sc2api_pb2 as sc_pb
from six import BytesIO as six_BytesIO
from starcraft_agents.bin import settings_interface
from starcraft_agents.deepq import learn
from sys import exit as sys_exit
from threading import Thread
from time import sleep
from time import time

# Needed for setup.py
def entry_point():
    app.run(main)

def get_game_version(replay_data):
    replay_io = six_BytesIO()
    replay_io.write(replay_data)
    replay_io.seek(0)
    archive = mpyq_MPQArchive(replay_io).extract()
    metadata = json_loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
    version = metadata["GameVersion"]
    return ".".join(version.split(".")[:-1])

def train_agent(agent_name,
                agent_race,
                agent2_race,
                cnn_to_mlp_args,
                difficulty,
                game_steps_per_episode,
                label,
                map_name,
                minimap_resolution,
                num_episodes,
                parallel,
                profile,
                replay_dir,
                save_replay,
                save_replay_episodes,
                screen_resolution,
                step_mul,
                trace,
                visualize):

    agent_module, agent_attribute = agent_name.rsplit(".", 1)
    agent_cls = getattr(import_module(agent_module), agent_attribute)
    file_dir = path.dirname(path.abspath(__file__))
    act_ops_dir = path.join(file_dir, "act_ops", map_name)
    makedirs(path.dirname(act_ops_dir), exist_ok=True)
    checkpoints_dir = path.join(file_dir, "checkpoints", map_name)
    makedirs(path.dirname(checkpoints_dir), exist_ok=True)
    tensorboard_dir = path.join(file_dir, "tensorboard", map_name)
    makedirs(path.dirname(tensorboard_dir), exist_ok=True)
    with sc2_env.SC2Env(map_name=map_name,
                        step_mul=step_mul,
                        visualize=visualize,
                        agent_interface_format=sc2_env.AgentInterfaceFormat(
                            feature_dimensions=sc2_env.Dimensions(
                                screen=64, minimap=64),
                                use_feature_units=True)) as env:
        train = learn.Learn(agent_cls,
                            cnn_to_mlp_args,
                            env,
                            map_name,
                            num_episodes,
                            save_replay)
        learn_kwargs = {'act_ops_dir': act_ops_dir,
                        'buffer_size': 10000,
                        'checkpoints_dir': checkpoints_dir,
                        'exploration_final_eps': 0.05,
                        'exploration_fraction': 0.8,
                        'learning_starts': 10000,
                        'prioritized_replay': False,
                        'target_network_update_freq': 1000,
                        'tensorboard_dir': tensorboard_dir,
                        'train_freq': 4}
        train(**learn_kwargs)
        with open(path.join(file_dir, "config.txt"), 'w') as file:
            file.write(json_dumps(learn_kwargs))

def main(_):
    if FLAGS.settings_interface:
        settings = settings_interface.TrainAgent()
        settings.master.mainloop()
    else:
        cnn_to_mlp_args = [[(32, 8, 4), (64, 4, 2), (64, 3, 1)], [256], FLAGS.dueling]
        train_agent(FLAGS.agent_name,
                    FLAGS.agent_race,
                    FLAGS.agent2_race,
                    cnn_to_mlp_args,
                    FLAGS.difficulty,
                    FLAGS.game_steps_per_episode,
                    FLAGS.label,
                    FLAGS.map_name,
                    FLAGS.minimap_resolution,
                    FLAGS.num_episodes,
                    FLAGS.parallel,
                    FLAGS.profile,
                    FLAGS.replay_dir,
                    FLAGS.save_replay,
                    FLAGS.save_replay_episodes,
                    FLAGS.screen_resolution,
                    FLAGS.step_mul,
                    FLAGS.trace,
                    FLAGS.visualize)

def play():
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    if (FLAGS.map_name and FLAGS.replay) or (not FLAGS.map_name and not FLAGS.replay):
        sys_exit("Must supply either a map or replay.")

    if FLAGS.replay and not FLAGS.replay.lower().endswith("sc2replay"):
        sys_exit("Replay must end in .SC2Replay.")

    if FLAGS.realtime and FLAGS.replay:
        sys_exit("Realtime isn't possible for replays yet.")

    if FLAGS.visualize and (FLAGS.realtime or FLAGS.full_screen):
        sys_exit("Disable pygame rendering if you want realtime or full_screen.")

    if platform_system() == "Linux" and (FLAGS.realtime or FLAGS.full_screen):
        sys_exit("Realtime and full_screen only make sense on Windows/MacOS.")

    if not FLAGS.visualize and FLAGS.render_sync:
        sys_exit("Render_sync only makes sense with pygame rendering on.")

    run_config = run_configs.get()
    # pylint: disable=E1101
    interface = sc_pb.InterfaceOptions()
    interface.raw = FLAGS.visualize
    interface.score = True
    interface.feature_layer.width = 24
    interface.feature_layer.resolution.x = FLAGS.screen_resolution
    interface.feature_layer.resolution.y = FLAGS.screen_resolution
    interface.feature_layer.minimap_resolution.x = FLAGS.minimap_resolution
    interface.feature_layer.minimap_resolution.y = FLAGS.minimap_resolution
    # pylint: enable=E1101

    max_episode_steps = FLAGS.game_steps_per_episode

    if FLAGS.map_name:
        map_inst = maps.get(FLAGS.map_name)
        if map_inst.game_steps_per_episode:
            max_episode_steps = map_inst.game_steps_per_episode
        create = sc_pb.RequestCreateGame(realtime=FLAGS.realtime,
                                         disable_fog=FLAGS.disable_fog,
                                         local_map=sc_pb.LocalMap(map_path=map_inst.path,
                                                                  map_data=map_inst.data(run_config)))
        create.player_setup.add(type=sc_pb.Participant)    # pylint: disable=E1101
        create.player_setup.add(type=sc_pb.Computer,    # pylint: disable=E1101
                                race=sc2_env.Race[FLAGS.agent2_race],
                                difficulty=sc2_env.Difficulty[FLAGS.difficulty])
        join = sc_pb.RequestJoinGame(race=sc2_env.Race[FLAGS.agent_race], options=interface)
        game_version = None
    else:
        replay_data = run_config.replay_data(FLAGS.replay)
        start_replay = sc_pb.RequestStartReplay(replay_data=replay_data,
                                                options=interface,
                                                disable_fog=FLAGS.disable_fog,
                                                observed_player_id=FLAGS.observed_player)
        game_version = get_game_version(replay_data)

    with run_config.start(game_version, FLAGS.full_screen) as controller:
        if FLAGS.map_name:
            controller.create_game(create)
            controller.join_game(join)
        else:
            info = controller.replay_info(replay_data)
            print("Replay info ".center(60, "-"))
            print(info)
            print("-" * 60)
            map_path = FLAGS.map_path or info.local_map_path
            if map_path:
                start_replay.map_data = run_config.map_data(map_path)
            controller.start_replay(start_replay)
        if FLAGS.visualize:
            renderer = renderer_human.RendererHuman(FLAGS.fps, FLAGS.step_mul, FLAGS.render_sync)
            renderer.run(run_config, controller, FLAGS.max_game_steps, max_episode_steps, FLAGS.save_replay)
        else:    # still step forward so the Mac/Windows renderer works
            try:
                while True:
                    frame_start_time = time()
                    if not FLAGS.realtime:
                        controller.step(FLAGS.step_mul)
                    obs = controller.observe()
                    if obs.player_result:
                        break
                    sleep(max(0, frame_start_time + 1 / FLAGS.fps - time()))
            except KeyboardInterrupt:
                pass
            print("Score: ", obs.observation.score.score)
            print("Result: ", obs.player_result)
            _, agent_name = FLAGS.agent_name.rsplit(".", 1)
            if FLAGS.map_name and FLAGS.save_replay:
                replay_save_loc = run_config.save_replay(controller.save_replay(), FLAGS.map_name, agent_name)
                print("Replay saved to:", replay_save_loc)
                # Save scores so we know how the human player did.
                with open(replay_save_loc.replace("SC2Replay", "txt"), "w") as file:
                    file.write("{}\n".format(obs.observation.score.score))

    if FLAGS.profile:
        print(stopwatch.sw)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("deepq_training", False, "Toggle DeepQ training.")
    # flags.DEFINE_bool("disable_fog", False, "Toggle fog of war.")
    flags.DEFINE_bool("dueling", True, "Toggle dueling networks for DeepQ training.")
    flags.DEFINE_bool("full_screen", False, "Toggle full screen.")
    flags.DEFINE_bool("prioritized_replay", False, "Toggle prioritized replay for DeepQ training.")
    # flags.DEFINE_bool("profile", False, "Toggle code profiling.")
    flags.DEFINE_bool("realtime", False, "Toggle realtime mode.")
    flags.DEFINE_bool("render_sync", False, "Toggle synchronized rendering.")
    # flags.DEFINE_bool("save_replay", True, "Toggle save replay.")
    flags.DEFINE_bool("settings_interface", False, "Toggle settings interface.")
    # flags.DEFINE_bool("trace", False, "Toggle code execution. tracing.")
    flags.DEFINE_bool("visualize", False, "Toggle pygame visualization.")
    # flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_, "Agent 1's race.")    # pylint: disable=protected-access
    # flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_, "Agent 2's race.")    # pylint: disable=protected-access
    # flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_, "If agent2 is a built-in Bot, it's strength.")    # pylint: disable=protected-access
    flags.DEFINE_float("fps", 22.4, "Frames per second to run the game.")
    flags.DEFINE_integer("experiment_num", 1, "Experiment number label.")
    # flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
    flags.DEFINE_integer("max_game_steps", None, "Total game steps to run.")
    flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
    flags.DEFINE_integer("num_episodes", 10, "Number of episodes.")
    flags.DEFINE_integer("observed_player", 1, "Which player to observe.")
    # flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
    flags.DEFINE_integer("save_replay_episodes", 0, "Save a replay after this many episodes.")
    flags.DEFINE_integer("screen_resolution", 84, "Resolution for screen feature layers.")
    # flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
    flags.DEFINE_string("agent_name", "pysc2.agents.random_agents.RandomAgent001", "Which agent to run.")
    flags.DEFINE_string("label", "starcraft-a", "Label for the experiment.")
    flags.DEFINE_string("map_name", None, "Name of a map to use.")
    flags.DEFINE_string("map_path", None, "Override the map for replay.")
    flags.DEFINE_string("replay", None, "Name of a replay to show.")
    flags.DEFINE_string("replay_dir", None, "Directory to save replays.")
    FLAGS.deepq_training = True
    FLAGS.map_name = "MoveToBeacon"
    FLAGS.map = "MoveToBeacon"
    FLAGS.agent_name = "starcraft_agents.minigame_agents.move_to_beacon_agents.MoveToBeaconAgent002"
    FLAGS.save_replay = False
    FLAGS.label = "starcraft-duel-a"
    app.run(main)
