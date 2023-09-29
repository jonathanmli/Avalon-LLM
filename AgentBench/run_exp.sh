# Assassin - w/ Discussion w/o Thought
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_local_llm.yaml --config configs/avalon_experiment/discussion_no_thought.yaml

# Assassin - w/ Discussion w/ Thought
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_local_llm.yaml --config configs/avalon_experiment/discussion_and_thought.yaml

<<<<<<< HEAD
# # All LLM - w/ Discussion w/ Thought Player 0 as Assassin
# python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/all_llm.yaml --config configs/avalon_experiment/all_llm.yaml

# # Assassin - w/o Discussion w/ Thought
# python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_assassin.yaml --config configs/avalon_experiment/no_discussion_w_thought.yaml

# Servant - w/o Discussion w/ Thought
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_assassin.yaml --config configs/avalon_experiment/servant_nd_wt.yaml
=======
# Assassin - w/o Discussion w/o Thought
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_local_llm.yaml --config configs/avalon_experiment/no_discussion_no_thought.yaml
>>>>>>> 4ba6b383ffe973517e9f63290c3050ae57682766
