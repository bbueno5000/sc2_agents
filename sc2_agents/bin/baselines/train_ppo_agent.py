from absl import app
from baselines import deepq
from collections import defaultdict
from gym import make
from gym_sc2 import envs

def callback(lcl, _glb):
    return lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199

def main(argv):
    env = make('MoveToBeacon-bbueno5000-v0')
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback)
    print('Saving model to move_to_beacon_deepq_model.pkl')
    act.save('move_to_beacon_deepq_model.pkl')
    experiment_to_name = {
        'atari-a': "Double Q learning",
        'atari-duel-a' : "Dueling Double Q learning",
        'atari-prior-a': "Double Q learning with Prioritized Replay",
        'atari-prior-duel-a' : "Dueling Double Q learning with Prioritized Replay",
        'atari-rb100-test': "Double Q learning with Replay buffer size = 100",
        'atari-rb10000-test': "Double Q learning with Replay buffer size = 10000",
        'atari-rb100000-test': "Double Q learning with Replay buffer size = 100000"}
    MAX_TSTEPS = int(2e8)
    experiments = sorted(experiment_to_name.keys())
    game_data = defaultdict(lambda: defaultdict(lambda: []))
    for run_name, data in run_to_episode_data.items():
        for experiment in experiments:
            if run_name.startswith(experiment):
                game = data['env_id'][:-len('NoFrameskip-v3')]
                t = np.cumsum(data['episode_data']["episode_lengths"])
                r = np.array(data['episode_data']["episode_rewards"])
                # Ensure all mesurements after the deadline of 2e8 are thrown away
                t_fltr, r_fltr = t[t < MAX_TSTEPS], r[t < MAX_TSTEPS]
                game_data[game][experiment].append((t_fltr, r_fltr))
                break

if __name__ == '__main__':
    app.run(main)
