# From Text to Tactic: Evaluating LLMs Playing the Game of Avalon

This is the official code for paper [From Text to Tactic: Evaluating LLMs Playing the Game of Avalon](https://browse.arxiv.org/pdf/2310.05036.pdf). The code is based on [AgentBench](https://github.com/THUDM/AgentBench).

## Initial Results

### LLMs Play Against Baseline Bots

Here are the results of LLMs playing against baseline bots.

![](./assets/sinlge_results.png)


### Multi-LLMs Self-Play

We also let LLMs playing against each other. Evil has an 8:2 advantage over Good, which is similar to the stats of rookie human players! Here are also some examples of discussion under this setting.

![](./assets/discussion1.png)

![](./assets/discussion2.png)


## Getting Started

### Prerequisites

`pip install -r requirements.txt`

### Installing

### Unit tests

To ensure that the code for the engine works, run the following from the root directory:
`python -m unittest discover Avalon`

## Running the experiments

- Run single-player setting with LLM playing as Assassin (w/ discussion)
```bash
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/single_player.yaml\
    --config configs/avalon_experiment/assassin_discussion.yaml
```

- Run single-player setting with LLM playing as Servant (w/ discussion)
```bash
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/single_player.yaml\
    --config configs/avalon_experiment/servant_discussion.yaml
```

- Run multi-player setting (w/ discussion)
```bash
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/all_llm.yaml\
    --config configs/avalon_experiment/all_llm.yaml
```

## Configuration

## Using the game engine

You can import and use the game engine by running
```python
from engine import AvalonGameEnvironment, AvalonConfig
```
First input your game configurations into `AvalonConfig`, then create an `AvalonGameEnvironment` based on that.

For an example of how to use the game engine, see `Avalon/test_engine.py`

## Authors

## License

