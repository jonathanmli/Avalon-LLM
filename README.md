# From Text to Tactic: Evaluating LLMs Playing the Game of Avalon

This is the official code for paper [From Text to Tactic: Evaluating LLMs Playing the Game of Avalon](TBD). The code is based on [AgentBench]().

## Getting Started

### Prerequisites

`pip install -r requirements.txt`

### Installing

### Unit tests

## Running the experiments

```bash
# Run single-player setting with LLM playing as Assassin (w/ discussion)
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/single_player.yaml --config configs/avalon_experiment/assassin_discussion.yaml
```

```bash
# Run single-player setting with LLM playing as Servant (w/ discussion)
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/single_player.yaml --config configs/avalon_experiment/servant_discussion.yaml
```

```bash
# Run multi-player setting (w/ discussion)
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/all_llm.yaml --config configs/avalon_experiment/all_llm.yaml
```

## Configuration

## Using the game engine

## Authors

## License

