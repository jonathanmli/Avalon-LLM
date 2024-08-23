import unittest
from strategist.searchlight.gameplay.simulators import DialogueGameSimulator
from strategist.Avalon.baseline_models_Avalon import *
from strategist.Avalon.engine import AvalonBasicConfig, AvalonGameEnvironment
from strategist.searchlight.gameplay.agents import MCTSAgent, MuteMCTSAgent, HumanDialogueAgent
from strategist.searchlight.classic_models import RandomRolloutValueHeuristic, ZeroValueHeuristic

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

# create 1 human agent, and fill the rest with random agents
rng = np.random.default_rng(12)
human_agents = dict()
for player in list(range(avalon_config.num_players)):
    if player == 0:
        human_agents[player] = HumanDialogueAgent(player, action_parser, rng)
    else:
        human_agents[player] = RandomDialogueAgent(rng)

# simulate games
num_games = 1
avg_scores, trajectories = simulator.simulate_games(human_agents, num_games, display=False)