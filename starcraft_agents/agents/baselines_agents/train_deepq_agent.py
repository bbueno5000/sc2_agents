import absl.app as app
import baselines.deepq as deepq
import gym
import gym_sc2

def callback(lcl, _glb):
    return lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199

def main(argv):
    env = gym.make('MoveToBeacon-bbueno5000-v0')
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
        callback=callback
        )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")

if __name__ == '__main__':
    app.run(main)
