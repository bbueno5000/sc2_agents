from absl import flags
from argparse import ArgumentParser
from move_to_beacon_1d import MoveToBeacon1d

FLAGS = flags.FLAGS
FLAGS([__file__])

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--visualize',
        type=bool,
        default=False,
        help='show the pysc2 visualizer')
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='number of episodes to run')
    parser.add_argument(
        '--step-mul',
        type=int,
        default=None,
        help='number of game steps to take per turn')
    args = parser.parse_args()
    example = MoveToBeacon1d(args.visualize, args.step_mul)
    rewards = example.run(args.num_episodes)
    print('Total reward: {}'.format(rewards.sum()))
    print('Average reward: {} +/- {}'.format(rewards.mean(), rewards.std()))
    print('Minimum reward: {}'.format(rewards.min()))
    print('Maximum reward: {}'.format(rewards.max()))

if __name__ == "__main__":
    main()
