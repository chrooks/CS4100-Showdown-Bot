#!/bin/bash

# Set the testing phase
TESTING_PHASE="higher_learn_rate"

# Set the number of runs
NUM_RUNS=3

# Run the Python script multiple times with different TESTING_ITER
for i in $(seq 1 $NUM_RUNS); do
    # Set the TESTING_ITER for the current run
    TESTING_ITER="$i" # This will be '1', '2', '3', etc.

    # Create the directory for the current run's results
    RESULTS_DIR="./results/${TESTING_PHASE}/${TESTING_ITER}"
    mkdir -p "$RESULTS_DIR"

    # Run the Python script with the environment variables set
    TESTING_PHASE="$TESTING_PHASE" TESTING_ITER="$TESTING_ITER" python showdown_rl_trainer.py
done

# Run calc_phase_averages with TESTING_PHASE set
TESTING_PHASE="$TESTING_PHASE" NUM_RUNS=3 python calc_phase_averages.py
