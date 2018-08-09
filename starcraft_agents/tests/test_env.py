import absl.app as app
import gym
import gym_sc2


def main(argv):
    env = gym.make('SC2MoveToBeacon-bbueno5000-v0')
    print(env.observation_space)
    print(env.action_space)
    Box(1, 64, 64)
    Discrete(4095)

if __name__ == '__main__':
    app.run(main)
