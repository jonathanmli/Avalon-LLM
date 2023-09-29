# Assassin - w/ Discussion w/o Thought
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_local_llm.yaml --config configs/avalon_experiment/discussion_no_thought.yaml

# Assassin - w/ Discussion w/ Thought
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_local_llm.yaml --config configs/avalon_experiment/discussion_and_thought.yaml

# Assassin - w/o Discussion w/o Thought
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_local_llm.yaml --config configs/avalon_experiment/no_discussion_no_thought.yaml