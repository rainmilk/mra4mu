#!/bin/bash

# Find and kill all running main_forget.py processes
pids=$(pgrep -f main_forget.py)

if [ -z "$pids" ]; then
    echo "No main_forget.py processes found."
else
    echo "Killing the following main_forget.py processes:"
    echo "$pids"
    kill -9 $pids
    echo "Processes have been killed."
fi
