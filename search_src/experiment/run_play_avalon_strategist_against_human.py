import unittest
from search_src.searchlight.gameplay.simulators import DialogueGameSimulator
from search_src.Avalon.baseline_models_Avalon import *
from search_src.Avalon.engine import AvalonBasicConfig, AvalonGameEnvironment
from search_src.dialogue_improve.action_planner import AvalonActionPlannerAgent
from search_src.searchlight.gameplay.agents import HumanDialogueAgent
from search_src.searchlight.classic_models import RandomRolloutValueHeuristic, ZeroValueHeuristic
from search_src.searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from good_examples.Avalon.value_heuristics.list import functions as value_heuristics
from good_examples.Avalon.dialogue_guide.list import guides as dialogue_guides

# create config
avalon_config = AvalonBasicConfig.from_num_players(5)
avalon_env = AvalonGameEnvironment(avalon_config)
start_state = AvalonState.init_from_env(avalon_env)

actor_action_enumerator = AvalonActorActionEnumerator(avalon_env=avalon_env)
forward_transitor = AvalonTransitor(env=avalon_env)
speaker_enumerator = AvalonSpeakerEnumerator(avalon_env=avalon_env)
information_function = AvalonInformationFunction(config=avalon_config)
action_parser = AvalonActorActionEnumerator.parse_str_to_action

# create game simulator
simulator = DialogueGameSimulator(transitor=forward_transitor, actor_action_enumerator=actor_action_enumerator, speaker_enumerator=speaker_enumerator, information_function=information_function, start_state=start_state)

# create inputs to AvalonActionPlannerAgent
llm_model = GPT35Multi(model="gpt-4")
value_heuristic = AvalonLLMFunctionalValueHeuristic(value_heuristics[0])
# dialogue_guide = dialogue_guides[0]

# create 1 human agent, and fill the rest with random agents
rng = np.random.default_rng(12)
human_agents = dict()
for i, player in enumerate(list(range(avalon_config.num_players))):
    if player == 0:
        human_agents[player] = HumanDialogueAgent(player, action_parser, rng)
    else:
        dialogue_guide = dialogue_guides[i % len(dialogue_guides)]
        role_to_dialogue_guide = {role: dialogue_guide for role in avalon_env.roles}
        human_agents[player] = AvalonActionPlannerAgent(config=avalon_config, llm_model=llm_model, player=player, value_heuristic=value_heuristic, role_to_dialogue_guide=role_to_dialogue_guide, rng=rng)
# print(human_agents[-1])

# simulate games
num_games = 1
avg_scores, trajectories = simulator.simulate_games(human_agents, num_games, display=False)