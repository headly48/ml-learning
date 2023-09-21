import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv


from flatland.utils.rendertools import RenderTool
from IPython.display import clear_output, display
import PIL

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO

import supersuit as ss

from gif_callback import GifCallback

# Rail Generator:

num_cities = 5  # Number of cities to place on the map
seed = 997  # Random seed
max_rails_between_cities = 2  # Maximum number of rails connecting 2 cities
max_rail_pairs_in_cities = 2  # Maximum number of pairs of tracks within a city
                              # Even tracks are used as start points, odd tracks are used as endpoints)

rail_generator = sparse_rail_generator(
    max_num_cities = num_cities,
    seed = seed,
    max_rails_between_cities = max_rails_between_cities,
    max_rail_pairs_in_city = max_rail_pairs_in_cities,
)


# Line Generator

# sparse_line_generator accepts a dictionary which maps speeds to probabilities.
# Different agent types (trains) with different speeds.
speed_probability_map = {
    1.    : 0.25,  # Fast passenger train
    1./ 2.: 0.25,  # Fast freight train
    1./ 3.: 0.25,  # Slow commuter train
    1./ 4.: 0.25   # Slow freight train
}

line_generator = sparse_line_generator(speed_probability_map)


stochastic_data = MalfunctionParameters(
    malfunction_rate = 1/10000,  # Rate of malfunction occurence
    min_duration = 15,  # Minimal duration of malfunction
    max_duration = 50  # Max duration of malfunction
)

malfunction_generator = ParamMalfunctionGen(stochastic_data)


# Observation Builder

# tree observation returns a tree of possible paths from the current position.
max_depth = 2  # Max depth of the tree
predictor = ShortestPathPredictorForRailEnv(max_depth=50)  # (Specific to Tree Observation - read code)

observation_builder = TreeObsForRailEnv(
    max_depth = max_depth, 
    predictor = predictor
)



from flatland.envs.observations import TreeObsForRailEnv

class SingleAgentNavigationObs(TreeObsForRailEnv):
    """
    We derive our observation builder from TreeObsForRailEnv, to exploit the existing implementation to compute
    the minimum distances from each grid node to each agent's target.

    We then build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """
    def __init__(self):
        super().__init__(max_depth=0)
        # We set max_depth=0 in because we only need to look at the current
        # position of the agent to decide what direction is shortest.

    def reset(self):
        # Recompute the distance map, if the environment has changed.
        super().reset()

    def get(self, handle):
        # Here we access agent information from the environment.
        # Information from the environment can be accessed but not changed!
        agent = self.env.agents[handle]

        possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = self._new_position(agent.position, direction)
                    min_distances.append(self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1

        return observation



width = 40  # Width of the map
height = 40  # Height of the map
number_of_agents = 3  # Number of trains to create
seed = 997  # Random seed

env = RailEnv(
    width = width,
    height = height,
    rail_generator = rail_generator,
    line_generator = line_generator,
    number_of_agents = number_of_agents,
    random_seed = seed,
    obs_builder_object = SingleAgentNavigationObs(),
    malfunction_generator = malfunction_generator
)




# Running the environment

env.reset()
# env = ss.concat_vec_envs_v1(env, 6, num_cpus=6, base_class='stable_baselines3')

callback = GifCallback()


model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
model.learn(total_timesteps=4000000, callback=callback)
model.save('flatland3-v1')











# step = 0
# while True:
#     step += 1

#     # actions is a dictionary with an action for each agent
#     actions = {}
#     for agent_i in range(len(env.agents)):
#         actions[agent_i] = np.random.randint(0, 5)

#     # env.step() accepts a dictionary of actions
#     next_observations, rewards, dones, info = env.step(actions)

#     # render_env(env)

#     # The reward setting is sparse anyways so print only the first 10 rewards
#     # rewards are only provided at the end of an episode
#     if (step < 10):
#         print("STEP: ", step, "\tREWARDS: ", rewards)

#     # break if all the agents are done or max episode steps is achieved
#     if dones["__all__"]:
#         print(".\n.\nSTEP: ", step, "\tREWARDS: ", rewards)
#         break


