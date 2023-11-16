# GOPS (Game of Pure Strategy)

This repo contains the code for the GOPS game.

## Quick start

### Start the task server and the assigner

Start the game for a naive game (both players takes the `random` strategy; 3 is the number of workers)
```bash
python -m src.start_task -a --start gops-dev-naive 3
```
Start the assigner
```bash
python -m src.assigner --config ./configs/assignments/test_gops.yaml
```

### Customize configurations and data

You can modify the file `configs/tasks/gops.yaml` to configure the agent list. A config file looks like this:
```yaml
default:
  module: "src.server.tasks.GOPS.GOPSBench"
  parameters:
    concurrency: 5
    num_games: 3
    num_turns: 5

gops-dev-naive:
  parameters:
    name: "gops-dev-naive"
    agent_list: ["naive", "naive"]
```
where you can change the number of games `num_games`, and the number of turns in each game `num_turns`. Advanced agents will be coming soon.
