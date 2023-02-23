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
  python3 /home/user/QtableV1/scrpits/learning_node.py --n_actions_enable 4 --resume --max_episodes 10 --exploration_func 2 --GOAL_POSITION 3 2 0  --log_file_dir "/home/user/QtableV1/Data/Log_learning" --Q_source_dir "/home/user/QtableV1/Data/Log_learning/Qtable.csv" --GOAL_RADIUS .5 --reward_threshold -600
done