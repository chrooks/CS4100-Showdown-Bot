"""
This module provides functionality to calculate and aggregate the average 
performance metrics across multiple iterations of reinforcement learning 
testing phases. It reads results from specified files and computes the 
overall averages for various performance indicators.

Functions:
    parse_results(file_path): Extracts performance metrics from a given results file.
    calculate_average_results(phase): Calculates and writes the average performance metrics 
                                      for all iterations in a given testing phase.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def parse_results(file_path: str):
    """
    Parses a results file to extract average performance, standard deviation,
    and weighted average performance.

    Args:
        file_path (str): The path to the results file.

    Returns:
        tuple: A tuple containing the average performance, standard deviation,
               and weighted average performance as floats.
    """
    avg_performance = 0
    std_dev = 0
    weighted_avg = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Iterate through each line to find the needed values
    for line in lines:
        if 'Weighted Average Performance:' in line:
            weighted_avg = float(line.split(':')[-1].strip())
        elif 'Average Performance:' in line:
            avg_performance = float(line.split(':')[-1].strip())
        elif 'Standard Deviation:' in line:
            std_dev = float(line.split(':')[-1].strip())

    return avg_performance, std_dev, weighted_avg


def calculate_average_results(phase: str):
    """
    Calculates the average performance metrics for all iterations within a testing phase.
    Writes the calculated averages to an output file named 'avg_results.txt' in the
    corresponding phase directory.

    Args:
        phase (str): The name of the testing phase.

    Outputs:
        A file 'avg_results.txt' in the phase directory containing the average metrics.
    """
    base_dir = "./results"
    phase_dir = os.path.join(base_dir, phase)
    total_avg_performance, total_std_dev, total_weighted_avg = 0, 0, 0

    for iteration in range(1, NUM_RUNS + 1):
        iteration_dir = os.path.join(phase_dir, str(iteration))
        results_file = os.path.join(iteration_dir, "results.txt")

        if os.path.exists(results_file):
            avg_performance, std_dev, weighted_avg = parse_results(
                results_file)
            print(
                f"From {results_file}: \
                \navg_perf: {avg_performance}, \
                \nstd_dev: {std_dev}, \
                \nweighted_avg: {weighted_avg}")
            total_avg_performance += avg_performance
            total_std_dev += std_dev
            total_weighted_avg += weighted_avg
        else:
            print(f"Results file not found: {results_file}")

    # Calculate averages
    avg_performance = total_avg_performance / NUM_RUNS
    avg_std_dev = total_std_dev / NUM_RUNS
    avg_weighted_performance = total_weighted_avg / NUM_RUNS

    # Write averages to file
    avg_results_file = os.path.join(phase_dir, "avg_results.txt")
    with open(avg_results_file, 'w', encoding='utf-8') as file:
        file.write(f"Average of Average Performance: {avg_performance:.4f}\n")
        file.write(f"Average of Standard Deviation: {avg_std_dev:.4f}\n")
        file.write(
            f"Average of Weighted Average Performance: {avg_weighted_performance:.4f}\n")

    print(f"Averages written to {avg_results_file}")


def smooth(y, box_pts):
    # Replace None values with 0 or use an appropriate method to handle them
    y_cleaned = [0 if x is None else x for x in y]

    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y_cleaned, box, mode='same')
    return y_smooth


def plot_phase_training_curves(phase: str):
    phase_dir = os.path.join("./results", phase)
    metrics = ['loss', 'mae', 'mean_q', 'episode_reward', 'nb_episode_steps']
    aggregated_data = {metric: [] for metric in metrics}
    num_episodes = None

    # Aggregate data
    for _, dirs, _ in os.walk(phase_dir):
        for dir in dirs:
            test_dir_path = os.path.join(
                phase_dir, dir, 'training_results.json')
            with open(test_dir_path, 'r') as file:
                data = json.load(file)
                if num_episodes is None:
                    num_episodes = len(data['episode'])
                for metric in metrics:
                    aggregated_data[metric].extend(data[metric])

    # Averaging data across all tests
    for metric in metrics:
        filtered_metric_data = [
            x for x in aggregated_data[metric] if x is not None]
        if len(filtered_metric_data) > 0:
            aggregated_data[metric] = [
                np.mean(filtered_metric_data[i::num_episodes]) for i in range(num_episodes)
            ]
        else:
            print(f"No valid data found for metric {metric}")

    # Plotting function
    def plot_metric(metric, title, smooth_factor=10):
        plt.figure()
        plt.plot(range(num_episodes),
                 aggregated_data[metric], label=metric, alpha=0.3)
        plt.plot(range(num_episodes), smooth(
            aggregated_data[metric], smooth_factor), label=f"Smoothed {metric}", linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel(metric)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(phase_dir, f"{metric}_curve.png"))

    # Plot different metrics
    for metric in metrics:
        plot_metric(metric, f"{metric.capitalize()} Over Episodes")


# Example usage
TESTING_PHASE = os.getenv('TESTING_PHASE', 'baseline')
NUM_RUNS = int(os.getenv('NUM_RUNS', '3'))
calculate_average_results(TESTING_PHASE)
print()
plot_phase_training_curves(TESTING_PHASE)
