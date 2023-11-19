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
where you can change the number of games `num_games`, and the number of turns in each game `num_turns`. We offer a basic prompt-based LLM agent now! To run `LLM vs. Random`, you should start the task server with
```bash
python -m src.start_task -a --start gops-dev-single 3
```

### Initial Results

#### LLM (gpt-3.5-turbo) vs. Random

```json
{
    "total": 10,
    "validation": {
        "running": 0.0,
        "completed": 1.0,
        "agent context limit": 0.0,
        "agent validation failed": 0.0,
        "agent invalid action": 0.0,
        "task limit reached": 0.0,
        "unknown": 0.0,
        "task error": 0.0,
        "average_history_length": 22.0,
        "max_history_length": 22,
        "min_history_length": 22
    },
    "custom": {
        "winrate of player 1": 0.5,
        "winrate of player 2": 0.5,
        "tie rate": 0.0
    }
}
```