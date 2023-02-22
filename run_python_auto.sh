#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <n>"
  exit 1
fi

# Assign the arguments to variables
N=$1

# Run the Python file n times
for ((i=1; i<=$N; i++)); do
  echo "Running iteration $i"
  python3 Qtable/QtableV1/scrpits/learning_node.py --max_episodes 15 --n_actions_enable 4 --GOAL_POSITION 3 -2 90 --resume
done



