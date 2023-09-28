# # Assassin - No Discussion No Thought
# python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_assassin.yaml --config configs/avalon_experiment/no_discussion_no_thought.yaml
# # Servant - w/ Discussion w/ Thought
# python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_assassin.yaml --config configs/avalon_experiment/servant_wd_wt.yaml

# # Servant - w/ Discussion w/o Thought
# python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_assassin.yaml --config configs/avalon_experiment/servant_wd_nt.yaml
# # Servant - w/o Discussion w/o Thought
# python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/test_assassin.yaml --config configs/avalon_experiment/servant_nd_nt.yaml

# All LLM - w/ Discussion w/ Thought Player 0 as Assassin
python eval.py --task configs/tasks/avalon/dev.yaml --agent configs/agents/all_llm.yaml --config configs/avalon_experiment/all_llm.yaml
