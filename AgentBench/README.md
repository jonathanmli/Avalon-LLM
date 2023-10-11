# Code for AvalonBench in AgentBench

Based on **AgentBench**, we implement AvalonBench, which extends **AgentBench** to support **multi-agent** game play of Avalon.

The specific code for our task can found in `../Avalon/avalon_task/`. Code for the agents are shown in `../Avalon/avalon_agent/`. If you are familiar with **AgentBench**, you can also find the code in their corresponding positions in **AgentBench**. Specifically, prompts that we use can be found in `../Avalon/avalon_task/prompts.py` and `../Avalon/avalon_task/task.py`. You can also change the prompts in config files in `configs/avalon_experiment/*.yaml`.



## Quick Start

To quickly understand how the framework works, you can follow the instructions below to run a simple evaluation.

### Step 1. Clone this repo and run the following command to install the requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2. Verify that you have successfully installed the requirements by running the following command:

```bash
python eval.py \
    --task configs/tasks/example.yaml \
    --agent configs/agents/do_nothing.yaml
```

### Step 3. Run Example Assignment

> *HINT: Example Assigment is composed of `gpt-3.5-turbo` and `ExampleTask` defined in [`src/tasks/example_task.py`](./src/tasks/example_task.py).*

You need to fill your [OPENAI KEY](https://platform.openai.com/account/api-keys) in `configs/assignments/example.yaml` first.

```yaml
Authorization: Bearer <%% PUT-YOUR-OPENAI-KEY-HERE %%>
```

Then run the following command:

```bash
python create_assignment.py \
    --assignment configs/assignments/example.yaml
```

### Step 4. Run Scripts for AvalonBench

> Above is the test code for AgentBench. However, you can simply run the scripts for AvalonBench

Finally, run the following scripts for AvalonBench. After that, you can check your output in the `avalon/logs` folder.
- Run single-player setting with LLM playing as Assassin (w/ discussion)
```bash
python eval.py \
    --task configs/tasks/avalon/dev.yaml \
    --agent configs/agents/single_player.yaml \
    --config configs/avalon_experiment/assassin_discussion.yaml
```

- Run single-player setting with LLM playing as Servant (w/ discussion)
```bash
python eval.py \
    --task configs/tasks/avalon/dev.yaml \
    --agent configs/agents/single_player.yaml \
    --config configs/avalon_experiment/servant_discussion.yaml
```

- Run multi-player setting (w/ discussion)
```bash
python eval.py \
    --task configs/tasks/avalon/dev.yaml \
    --agent configs/agents/all_llm.yaml \
    --config configs/avalon_experiment/all_llm.yaml
```