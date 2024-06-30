from search_src.searchlight.utils import AbstractLogged
from ..Avalon.baseline_models_Avalon import AvalonState
from search_src.searchlight.gameplay.agents import MuteMCTSAgent
from search_src.searchlight.datastructures.graphs import PartialValueGraph
from search_src.searchlight.datastructures.adjusters import PUCTAdjuster
from search_src.searchlight.datastructures.estimators import UtilityEstimatorMean
from .dialogue_discrimination import DialogueDiscriminator
from .dialogue_generator import DialogueGenerator
from .prompt_generator import PromptGenerator as PPromptGenerator


import numpy as np
from collections import defaultdict
from search_src.Avalon.baseline_models_Avalon import *


import numpy as np


class AvalonActionPlannerAgent(MuteMCTSAgent):

    observed_dialogue: dict[tuple[int, int], list[tuple[int, str]]]
    role_to_dialogue_guide: dict[int, str]

    def __init__(self, config: AvalonBasicConfig, llm_model: LLMModel, player: int, value_heuristic: ValueHeuristic, role_to_dialogue_guide: dict[int, str], num_rollout: int = 100, node_budget: int = 100, rng: np.random.Generator = np.random.default_rng()):
        
        print("Initializing Action Planner Agent")
        # create new game environment for simulation so that we don't mess up the main game environment
        env = AvalonGameEnvironment.from_num_players(config.num_players)
        num_players = env.config.num_players
        forward_transitor = AvalonTransitor(env)
        actor_action_enumerator = AvalonActorActionEnumerator(env)
        policy_predictor = PolicyPredictor()
        information_function = AvalonInformationFunction(config=config)
        
        # start_state = AvalonState.init_from_env(env)
        players = set([i for i in range(num_players)])
        self.config = config
        self.observed_dialogue = defaultdict(list)
        self.dialogue_gudies = role_to_dialogue_guide

        self.dialogue_discriminator = DialogueDiscriminator(llm_model=llm_model, prompt_generator=PPromptGenerator(self.config), player=player, players=players,)
        information_prior = AvalonInformationPrior(config=config, belief_p_is_merlin=self.dialogue_discriminator.get_p_is_merlin(), belief_p_is_good=self.dialogue_discriminator.get_p_is_good(),)

        self.dialogue_generator = DialogueGenerator(llm_model=llm_model, prompt_generator=PPromptGenerator(self.config),)

        super(). __init__(players=players, player=player, forward_transitor=forward_transitor, actor_action_enumerator=actor_action_enumerator, value_heuristic=value_heuristic, policy_predictor=policy_predictor, information_function=information_function, information_prior=information_prior, num_rollout=num_rollout, node_budget=node_budget, rng=rng)

    def _observe_dialogue(self, state: AvalonInformationSet, new_dialogue: list[tuple[int, str]]):
        '''
        Observes new dialogue and updates internal states
        '''

        if not self.dialogue_discriminator.check_role_equivalent(state.self_role):
            # reset the dialogue discriminator if the role is not equivalent
            self.dialogue_discriminator.reset(known_sides=state.known_sides, player_role=state.self_role)

        # we need to first convert the state and new_dialogue to string form here before passing it to the discriminator
        state_string_description = state.gen_str_description()
        new_dialogue_string = self.dialogue_list_to_str(new_dialogue)
        full_description = f"""{state_string_description}
        
        This round, players have said the following so far:
        {new_dialogue_string}"""

        # print("Full description: ", full_description)
        self.dialogue_discriminator.update_beliefs(full_description)
        self.observed_dialogue[(state.turn, state.round)].extend(new_dialogue)
        self.information_prior = AvalonInformationPrior(config=self.config, belief_p_is_merlin=self.dialogue_discriminator.get_p_is_merlin(), belief_p_is_good=self.dialogue_discriminator.get_p_is_good(),)# TODO: we need to update the information prior here
    
    @staticmethod
    def dialogue_list_to_str(dialogue: list[tuple[int, str]]):
        return '\n'.join([f"Player {player} said: {dialogue}" for player, dialogue in dialogue])

    def _produce_utterance(self, state: AvalonInformationSet,) -> str:
        '''
        Produces a dialogue given a history
        '''

        # get the dialogue for this turn and round
        dialogue = self.observed_dialogue[(state.turn, state.round)]

        # we need to first convert the state and new_dialogue to string form here before passing it to the discriminator
        state_string_description = state.gen_str_description()
        new_dialogue_string = self.dialogue_list_to_str(dialogue) # NOTE: might need to add preamble
        full_description = f"""{state_string_description}
        
        This round, players have said the following so far:
        {new_dialogue_string}"""

        # query action planner for intended action
        # actor, actions = self.actor_action_enumerator.enumerate(state)
        intended_action = self._act(state, actions=set("not yet decided")) # NOTE: we might need to change this to actions
        tips = self.dialogue_gudies[state.self_role]

        response, notes = self.dialogue_generator.generate_dialogue(history=full_description, phase=state.phase, intended_action=intended_action, tips=tips)
        return response

    # def _act(self, information_set: AvalonInformationSet, actions: set[Hashable]):
        