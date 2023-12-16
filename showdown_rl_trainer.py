"""
This module sets up a reinforcement learning environment using the Pokémon Showdown
game environment. It defines a Dueling Deep Q-Network (DQN) agent, trains it using 
the specified configurations, and evaluates its performance against various opponents 
in the Pokémon battle simulator. The module integrates various components such as 
custom environments, processors, and memory strategies for the DQN agent.
"""
import asyncio
import os
import numpy as np
import matplotlib.pyplot as plt
from gym.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tabulate import tabulate
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers.legacy import Adam
from matplotlib import ticker
from poke_env.player import (
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    wrap_for_old_gym_api,
)
from utils import (
    SAVE_TO_FILE,
    CustomEnvWrapper,
    CustomProcessor,
    CustomFileLogger,
    DuelingQLayer,
    save_or_print,
)
from dojo import Dojo


BATTLE_FORMAT = "gen8randombattle"
TRAINING_STEPS = 10000
EVAL_CHALLENGES = 50

SAVE_TO_FILE = False
RESULT_FPATH = "./results"

# Use environment variables if they exist
TESTING_PHASE = os.getenv('TESTING_PHASE', "test")
TESTING_ITER = os.getenv('TESTING_ITER', "1")

RESULTS_DIR = os.path.join(RESULT_FPATH, TESTING_PHASE, TESTING_ITER)
if SAVE_TO_FILE and not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
RESULTS_FPATH = os.path.join(RESULTS_DIR, 'results.txt')
TRAINING_RESULTS_FPATH = os.path.join(RESULTS_DIR, 'training_results.json')


async def main():
    """
    The main coroutine of the script. This function sets up the environment, trains 
    a Dueling DQN agent in the Pokémon Showdown environment, and evaluates its 
    performance against a set of opponents.

    The training involves the following steps:
    - Setting up test, training, and evaluation environments.
    - Creating a neural network model for the agent.
    - Configuring the agent with memory, policy, and other training parameters.
    - Training the agent in the environment.
    - Evaluating the agent's performance through cross-evaluation.
    - Plotting and saving the results if specified.

    Raises:
        ConnectionRefusedError: If the Pokémon Showdown server connection fails.

    Outputs:
        - Training and evaluation statistics.
        - Visualizations of performance metrics.
        - Saved results in a specified directory if enabled.
    """
    try:
        # Test environment setup for consistency with OpenAI Gym standards
        opponent = RandomPlayer(battle_format=BATTLE_FORMAT)
        test_env = Dojo(
            battle_format=BATTLE_FORMAT, start_challenging=True, opponent=opponent
        )
        check_env(test_env)  # Verify environment compatibility with Gym API
        test_env.close()  # Close the test environment

        # Create environments for training and evaluation
        # Training environment
        opponent = RandomPlayer(battle_format=BATTLE_FORMAT)
        train_env = Dojo(
            battle_format=BATTLE_FORMAT, opponent=opponent, start_challenging=True
        )
        train_env = CustomEnvWrapper(wrap_for_old_gym_api(
            train_env))  # Wrapping for RL compatibility

        # Evaluation environment
        opponent = RandomPlayer(battle_format=BATTLE_FORMAT)
        eval_env = Dojo(
            battle_format=BATTLE_FORMAT, opponent=opponent, start_challenging=True
        )
        eval_env = CustomEnvWrapper(wrap_for_old_gym_api(
            eval_env))  # Wrapping for RL compatibility
    except ConnectionRefusedError as e:
        print(f"Failed to connect to the server: {e}")
        print(
            "Make sure the local Pokémon Showdown server is running on the specified port.")
        return  # Exit early if connection fails

    # Determine the number of possible actions from the environment
    n_action = train_env.action_space.n

    # Define the input layer
    input_layer = Input(shape=(1, 882))

    # Common layers
    common = Dense(128, activation='elu')(input_layer)
    common = Flatten()(common)
    common = Dense(64, activation='elu')(common)

    # Split into two streams
    # State Value stream
    state_value = Dense(32, activation='elu')(common)
    state_value = Dense(1, activation='linear')(state_value)

    # Action Advantage stream
    action_advantage = Dense(32, activation='elu')(common)
    action_advantage = Dense(n_action, activation='linear')(action_advantage)

    # Combine state and advantage to get Q-values
    # Replace Lambda layer with your custom DuelingQLayer
    dueling_output = DuelingQLayer(dueling_type='avg')(
        [state_value, action_advantage])

    # Create the model
    model = Model(inputs=input_layer, outputs=dueling_output)

    # Configure the memory for experience replay in DQN
    memory = SequentialMemory(limit=TRAINING_STEPS, window_length=1)

    # Define exploration policy with linear annealing for balancing exploration and exploitation
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),  # Epsilon-greedy policy
        attr="eps",
        value_max=1.0,   # Start with high exploration
        value_min=0.05,  # Reduce exploration over time
        value_test=0.0,  # Exploration factor during testing
        nb_steps=TRAINING_STEPS,  # Total steps for annealing
    )

    # Initialize the DQN agent with the model, policy, and other configurations
    dqn = DQNAgent(
        model=model,                    # Q-network
        nb_actions=n_action,
        policy=policy,                  # Exploration policy
        memory=memory,                  # Memory for experience replay
        nb_steps_warmup=TRAINING_STEPS / 10,  # Warmup steps
        gamma=0.5,                      # Discount factor
        target_model_update=1,          # Target model update frequency
        delta_clip=0.01,                # Clipping delta
        enable_double_dqn=True,         # Enable Double DQN
        enable_dueling_network=False,    # Enable Dueling DQN
        dueling_type='avg',             # Dueling type
        # Include custom layer
        custom_model_objects={'DuelingQLayer': DuelingQLayer}
    )

    dqn.processor = CustomProcessor()  # Set custom processor for observation handling
    dqn.compile(Adam(learning_rate=0.001), metrics=[
                "mae"])  # Compile the DQN agent

    print()

    callbacks = []
    if SAVE_TO_FILE:
        file_logger = CustomFileLogger(
            TRAINING_RESULTS_FPATH, TRAINING_STEPS / 100)
        callbacks.append(file_logger)

    # Train the DQN agent on the training environment
    dqn.fit(train_env, nb_steps=TRAINING_STEPS,
            visualize=True, verbose=2, callbacks=callbacks)

    train_env.close()  # Close the training environment after training

    eval_env.reset_env(restart=False)  # Reset the evaluation environment

    # Perform cross-evaluation against various players
    n_challenges = EVAL_CHALLENGES  # Number of challenges for evaluation
    # List of players for cross evaluation, including the trained agent
    players = [
        eval_env.agent,
        RandomPlayer(battle_format=BATTLE_FORMAT),
        MaxBasePowerPlayer(battle_format=BATTLE_FORMAT),
        SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT),
    ]
    cross_eval_task = background_cross_evaluate(
        players, n_challenges)  # Start cross evaluation in background
    dqn.test(
        eval_env,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=True,
        visualize=False,
    )

    cross_evaluation = cross_eval_task.result()  # Get results of cross evaluation
    table = [["-"] + [p.username for p in players]]
    win_rates = []
    opponents = []
    for p_1, _ in cross_evaluation.items():
        row = [p_1]
        for p_2 in players:
            win_rate = cross_evaluation[p_1][p_2.username]
            row.append(win_rate)
            # Check if the player is the DQN agent and the win rate is not None
            if p_1 == eval_env.agent.username and win_rate is not None:
                win_rates.append(win_rate)
                opponents.append(p_2.username)  # Add opponent's username
        table.append(row)

    if win_rates:
        average_performance = np.mean(win_rates)
        performance_std_dev = np.std(win_rates)
        weights = {
            "RandomPlayer 4": 1,
            "MaxBasePowerPlay 1": 7.665994,
            "SimpleHeuristics 1": 128.757145
        }
        total_weighted_win_rate = 0
        total_weights = 0
        for opponent, win_rate in zip(opponents, win_rates):
            weight = weights.get(opponent, 0)
            total_weighted_win_rate += win_rate * weight
            total_weights += weight

        weighted_average_performance = total_weighted_win_rate / \
            total_weights if total_weights else 0

    # Open a file to write the results
    if SAVE_TO_FILE:
        with open(RESULTS_FPATH, 'w', encoding='utf-8') as file:
            save_or_print("\nCross evaluation of DQN with baselines:", file)
            save_or_print(tabulate(table), file)

            if win_rates:
                save_or_print(
                    f"\nAverage Performance: {average_performance:.3f}", file)
                save_or_print(
                    f"Standard Deviation: {performance_std_dev:.3f}\n", file)
                save_or_print(
                    f"Weighted Average Performance: {weighted_average_performance:.3f}\n", file)

    else:
        save_or_print("\nCross evaluation of DQN with baselines:", file)
        save_or_print(tabulate(table), file)

        if win_rates:
            save_or_print(
                f"\nAverage Performance: {average_performance:.3f}", file)
            save_or_print(
                f"Standard Deviation: {performance_std_dev:.3f}\n", file)
            save_or_print(
                f"Weighted Average Performance: {weighted_average_performance:.3f}\n", file)
            save_or_print(
                f"Weighted Average Performance: {weighted_average_performance:.3f}\n", file)


    # Plotting the bar chart for win rates
    plt.figure(figsize=(10, 6))
    plt.bar(opponents, win_rates, color='blue')
    plt.xlabel('Opponents')
    plt.ylabel('Win Rate')
    plt.title('Performance of DQN Agent Against Different Players')
    plt.ylim(0, max(1.0, *win_rates))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    if SAVE_TO_FILE:
        plt.savefig(os.path.join(RESULTS_DIR, 'win_rates_chart.png'))
    else:
        plt.show()



if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
