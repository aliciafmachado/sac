"""
Main file for passing the parameters and calling training.
"""

import argparse


parser = argparse.ArgumentParser(description='Training SAC agent.')
parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
args = parser.parse_args()


def __main__():
    # TODO: pass arguments to training function and call training of agent
    pass


if __name__ == "__main__":
    __main__()