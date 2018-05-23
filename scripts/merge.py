from __future__ import division, print_function
import multiprocessing
import gym

from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.trainer.simulation import Simulation


def main():
    gym.logger.set_level(gym.logger.INFO)
    env = gym.make('highway-merge-v0')
    agent = MCTSAgent(env,
                      prior_policy=MCTSAgent.fast_policy,
                      rollout_policy=MCTSAgent.idle_policy,
                      iterations=50,
                      assume_vehicle_type=None)
    sim = Simulation(env, agent, num_episodes=10)
    sim.test()


if __name__ == '__main__':
    for i in range(4):
        p = multiprocessing.Process(target=main)
        p.start()