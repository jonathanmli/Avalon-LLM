# Essential Code for AvalonBench

This directory maintains the essential code for AvalonBench. We separately put the code here for ease of secondary development.

## Engine

The game environment is in the file `engine.py`

## Baseline Bots

The code for our baseline bots is shown in `baseline_agents.py`

## Experiment for Baseline Bots

You can run the experiment of baseline bots playing against each other with the following line of code, setting `NUM_GAMES` argument to the desired number of games:

```bash
python naive_experiment.py --num_games NUM_GAMES
```

## Code for AgentBench

In `avalon_agent` and `avalon_task`, you will find the code for integrating **AvalonBench** into **AgentBench**. For secondary development, please refer to the [tutorial in AgentBench](https://github.com/THUDM/AgentBench/blob/main/docs/tutorial.md).

## Prompts

Prompts that we use can be found in `avalon_task/prompts.py` and `avalon_task/task.py` (which will also be merged into `avalon_task/prompts.py`).