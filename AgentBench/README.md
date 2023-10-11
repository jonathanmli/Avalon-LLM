![](./assets/cover.jpg)
<p align="center">
   <a href="https://llmbench.ai" target="_blank">🌐 Website</a> | <a href="https://twitter.com/thukeg" target="_blank">🐦 Twitter</a> | <a href="mailto:agentbench@googlegroups.com">✉️ Google Group</a> | <a href="https://arxiv.org/abs/2308.03688" target="_blank">📃 Paper </a>
</p>

<p align="center">
👋 Join our <a href="https://join.slack.com/t/agentbenchcol-huw1944/shared_invite/zt-20ixabcuv-31cFLBAkqGQxQkJqrWVEVg" target="_blank">Slack</a>  for <i>Q & A</i> or <i><b>collaboration</b> on AgentBench v2.0</i>!
</p>

# AgentBench: Evaluating LLMs as Agents

> This is a modified version of the original README in AgentBench. You can also follow this README to run AvalonBench

**AgentBench** is the first benchmark designed to evaluate **LLM-as-Agent** across a diverse spectrum of different environments. Based on that, we implement AvalonBench, which extends **AgentBench** to support **multi-agent** game play of Avalon.

The specific code for our task can found in `../Avalon/avalon_task/`. Code for the agents are shown in `../Avalon/avalon_agent/`. If you are familiar with **AgentBench**, you can also find the code in their corresponding positions in **AgentBench**.


## Table of Contents


- [Quick Start](#quick-start)
- [Citation](#citation)


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

> Above are the test code for AgentBench. However, you can simply run the scripts for AvalonBench

Finally, run the following scripts for AvalonBench. After that, you can check your output in the `logs` folder.
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

## Citation

```
@article{liu2023agentbench,
  title   = {AgentBench: Evaluating LLMs as Agents},
  author  = {Xiao Liu and Hao Yu and Hanchen Zhang and Yifan Xu and Xuanyu Lei and Hanyu Lai and Yu Gu and Hangliang Ding and Kaiwen Men and Kejuan Yang and Shudan Zhang and Xiang Deng and Aohan Zeng and Zhengxiao Du and Chenhui Zhang and Sheng Shen and Tianjun Zhang and Yu Su and Huan Sun and Minlie Huang and Yuxiao Dong and Jie Tang},
  year    = {2023},
  journal = {arXiv preprint arXiv: 2308.03688}
}
```
