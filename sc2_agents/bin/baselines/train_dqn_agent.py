from absl import app
from baselines import deepq
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
        max_timesteps=1000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print('Saving model to move_to_beacon_deepq_model.pkl')
    act.save('move_to_beacon_deepq_model_1.pkl')

if __name__ == '__main__':
    app.run(main)
