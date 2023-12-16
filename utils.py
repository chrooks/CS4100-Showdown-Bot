"""
This utility module provides essential classes and functions used in setting up and 
managing the reinforcement learning environment for Pokémon Showdown. It includes custom
processing for observations, environment wrappers for specific handling in the Pokémon 
context, and a custom Dueling Q Layer for neural network models.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gym.wrappers
from rl.callbacks import FileLogger
from keras.callbacks import Callback
from keras.layers import Layer
from rl.core import Processor

SAVE_TO_FILE = True


def save_or_print(text, file=None):
    """
    Outputs the given text either to a file or prints it to the console, based on 
    the global SAVE_TO_FILE flag.

    Args:
        text (str): The text to be outputted.
        file (Optional[io.TextIOWrapper]): The file object to write to, if applicable.
    """

    if SAVE_TO_FILE:
        print(text, file=file)
    else:
        print(text)


class CustomFileLogger(FileLogger):
    def save_data(self):
        """ Save metrics in a json file """
        if len(self.data.keys()) == 0:
            return

        # Sort everything by episode.
        assert 'episode' in self.data
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, _ in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            # We convert to np.array() and then to list to convert from np datatypes to native datatypes.
            # This is necessary because json.dump cannot handle np.float32, for example.
            sorted_data[key] = np.array(
                [self.data[key][idx] for idx in sorted_indexes]).tolist()
            sorted_data[key] = [None if np.isnan(x) else x for x in sorted_data[key]]

        # Overwrite already open file. We can simply seek to the beginning since the file will
        # grow strictly monotonously.
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f)


class CustomProcessor(Processor):
    """
    Custom processor for handling observations in a reinforcement learning environment.
    This processor adapts the observation handling to the specific requirements of 
    the Pokémon Showdown environment.

    Overrides:
        process_observation: Processes the observation from the environment. If the 
        observation is a tuple with two elements, it returns the first element. 
        Otherwise, returns the observation as is.
    """

    def process_observation(self, observation):
        # Check if observation is a tuple and has exactly 2 elements
        if isinstance(observation, tuple) and len(observation) == 2:
            # Unpack the observation from the tuple
            obs, _ = observation
            return obs
        # If it's not a tuple of size 2, return the observation as is
        return observation


class CustomEnvWrapper(gym.Wrapper):
    """
    Custom environment wrapper for the OpenAI Gym environment, specifically adapted 
    for the Pokémon Showdown environment. It modifies the step method to handle 
    actions and their results in the context of Pokémon battles.

    Overrides:
        step: Executes the given action in the environment and returns the new 
        observation, reward, and termination status along with any additional info.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        done = terminated or truncated
        return observation, reward, done, info


class DuelingQLayer(Layer):
    """
    Custom layer for implementing Dueling Q Network architecture in a neural network model.
    This layer computes the Q-value by combining state values and action advantages 
    based on a specified dueling type.

    Args:
        dueling_type (str): Type of dueling implementation ('avg', 'max', or 'naive').

    Overrides:
        call: Computes the output of the layer based on inputs and the dueling type.
        compute_output_shape: Returns the shape of the output tensor.
        get_config: Returns the configuration of the layer.
    """

    def __init__(self, dueling_type='avg', **kwargs):
        self.dueling_type = dueling_type
        super(DuelingQLayer, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        state_value = inputs[0]
        action_advantage = inputs[1]

        # Ensure action_advantage is 2D: (batch_size, num_actions)
        action_advantage = tf.expand_dims(
            action_advantage, axis=-1) if tf.rank(action_advantage) == 1 else action_advantage

        if self.dueling_type == 'avg':
            output = state_value + \
                (action_advantage - tf.reduce_mean(action_advantage, axis=1, keepdims=True))
        elif self.dueling_type == 'max':
            output = state_value + \
                (action_advantage - tf.reduce_max(action_advantage, axis=1, keepdims=True))
        elif self.dueling_type == 'naive':
            output = state_value + action_advantage
        else:
            raise ValueError("Invalid dueling type")
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super(DuelingQLayer, self).get_config()
        config['dueling_type'] = self.dueling_type
        return config


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_learning_curves(directory, save_to_file=False):
    # Load data
    file_path = os.path.join(directory, 'training_results.json')
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Plotting function
    def plot_metric(metric, title, smooth_factor=10):
        plt.figure()
        plt.plot(data['episode'], data[metric], label=metric, alpha=0.3)  # Original data with lower opacity
        plt.plot(data['episode'], smooth(data[metric], smooth_factor), label=f"Smoothed {metric}", linewidth=2)  # Smoothed data
        plt.xlabel('Episode')
        plt.ylabel(metric)
        plt.title(title)
        plt.legend()
        if save_to_file:
            plt.savefig(os.path.join(directory, f"{metric}_curve.png"))
        else:
            plt.show()

    # Plot different metrics
    plot_metric('loss', 'Loss Over Episodes')
    plot_metric('mae', 'MAE Over Episodes')
    plot_metric('mean_q', 'Mean Q-Value Over Episodes')
    plot_metric('episode_reward', 'Episode Reward Over Episodes')
    plot_metric('nb_episode_steps', 'Number of Steps per Episode')
