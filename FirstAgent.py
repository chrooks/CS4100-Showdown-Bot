import asyncio
import json

import numpy as np
import gym.wrappers
from gym.spaces import Box, Space
from gym.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tabulate import tabulate
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers.legacy import Adam

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data.gen_data import GenData
from poke_env.player import (
    Gen9EnvSinglePlayer,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
    wrap_for_old_gym_api,
)

# This script sets up a reinforcement learning environment using the Pokemon Showdown
# game environment. It defines a simple RL agent, trains it using Deep Q-Networks (DQN),
# and evaluates its performance against various opponents.


class SimpleRLPlayer(Gen9EnvSinglePlayer):
    # Calculate the reward for the current state of the game
    def calc_reward(self, _, current_battle):
        return self.reward_computing_helper(current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0)

    # In summary, embed_battle transforms current state of a Pokémon battle -> numerical format suitable for ML models.
    # Includes info about the available moves' base power and effectiveness & how many fainted pokemon there are on each side.
    def embed_battle(self, battle: AbstractBattle):
        gen_9_type_chart = GenData.from_gen(9).type_chart
        # Initialize arrays for move base power and damage multipliers
        # initial value of -1.0  indicates an uninitialized or unavailable move
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100  # Simple rescaling to facilitate learning
            )
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=gen_9_type_chart
                )

        # Count fainted Pokemon for both teams
        fainted_mon_team = len(
            [mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Combine all features into a single vector
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    # Define the structure and limits of the observation space
    def describe_embedding(self) -> Space:
        # [base_power[1-4], multiplier[1-4],fainted_mon_team, fainted_mon_opponent]
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


class CustomProcessor(Processor):
    def process_observation(self, observation):
        # Check if observation is a tuple and has exactly 2 elements
        if isinstance(observation, tuple) and len(observation) == 2:
            # Unpack the observation from the tuple
            obs, _ = observation
            return obs
        else:
            # If it's not a tuple of size 2, return the observation as is
            return observation


class CustomEnvWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        done = terminated or truncated
        return observation, reward, done, info


async def main():
    # Test the environment for consistency with OpenAI Gym API
    opponent = RandomPlayer(battle_format="gen9randombattle")
    test_env = SimpleRLPlayer(
        battle_format="gen9randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    # Create training and evaluation environments
    opponent = RandomPlayer(battle_format="gen9randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen9randombattle", opponent=opponent, start_challenging=True
    )
    train_env = CustomEnvWrapper(wrap_for_old_gym_api(train_env))

    opponent = RandomPlayer(battle_format="gen9randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen9randombattle", opponent=opponent, start_challenging=True
    )
    eval_env = CustomEnvWrapper(wrap_for_old_gym_api(eval_env))

    # Determine the action and observation space dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    # Define the neural network model for DQN
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10)))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Configure the DQN agent
    memory = SequentialMemory(limit=10000, window_length=1)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )
    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.processor = CustomProcessor()
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Train the DQN agent
    dqn.fit(train_env, nb_steps=10)
    train_env.close()

    # Evaluate the trained model against different opponents
    print("Results against random player:")
    dqn.test(eval_env, nb_episodes=5, verbose=False, visualize=True)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    second_opponent = MaxBasePowerPlayer(battle_format="gen9randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    dqn.test(eval_env, nb_episodes=5, verbose=False, visualize=True)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=True)

    # Use utility methods to evaluate the player
    n_challenges = 250
    placement_battles = 40
    eval_task = background_evaluate_player(
        eval_env.agent, n_challenges, placement_battles
    )
    dqn.test(eval_env, nb_episodes=n_challenges,
             verbose=False, visualize=True)
    print("Evaluation with included method:", eval_task.result())
    eval_env.reset_env(restart=True)

    # Perform cross-evaluation against various players
    n_challenges = 50
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen9randombattle"),
        MaxBasePowerPlayer(battle_format="gen9randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen9randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    dqn.test(
        eval_env,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=False,
        visualize=True,
    )
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
