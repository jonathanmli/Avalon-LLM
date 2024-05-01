from search_src.searchlight.headers import Any
from search_src.searchlightimprove.headers import Evaluator
from search_src.searchlight.gameplay.simulators import GameSimulator
from search_src.searchlight.headers import *
from search_src.searchlight.gameplay.agents import SearchAgent
from search_src.searchlight.algorithms.mcts_search import SMMonteCarlo
from search_src.searchlight.datastructures.graphs import ValueGraph2, PartialValueGraph
from search_src.searchlight.datastructures.adjusters import PUCTAdjuster
from search_src.searchlight.datastructures.estimators import UtilityEstimatorLast
from search_src.searchlight.classic_models import RandomRolloutValueHeuristic
from search_src.searchlightimprove.evaluators import SimulateSearchGameEvaluator
from search_src.searchlightimprove.llm_utils.llm_api_models import LLMModel

from .dialogue_discrimination import DialogueDiscriminator
from .dialogue_generator import DialogueGenerator
from .data_loader import DataLoader
from .prompt_generator import PromptGenerator
from ..Avalon.baseline_models_Avalon import AvalonState, AvalonBasicConfig



from typing import Type

class PromptSSGEvaluator(Evaluator):
    '''
    Evaluates the prompts using the discriminator
    '''

    def __init__(self, players: set, role_to_evaluate: int, data_loader: DataLoader, llm_model: LLMModel, prompt_generator: PromptGenerator, num_batch_runs: int = 10, rng: np.random.Generator = np.random.default_rng(),):
        '''
        Args:
            simulator: game simulator
            transitor: forward transitor
            actor_enumerator: actor enumerator
            action_enumerator: action enumerator
            players: set of players
            role_to_evaluate: role to evaluate
            num_batch_runs: number of evaluation runs
            rng: random number generator
            against_benchmark: whether to evaluate against benchmark
            search_budget: search budget
            random_rollouts: number of random rollouts
        '''
        super().__init__()
        self.role_to_evaluate = role_to_evaluate
        self.data_loader = data_loader
        self.llm_model = llm_model
        self.prompt_generator = prompt_generator
        self.num_batch_runs = num_batch_runs
        self.players = players
        self.rng = rng

    def evaluate(self, prompts: list[str]) -> tuple[list[float], list]:
        # evaluate each prompt
        scores = []
        notes = []
        for prompt in prompts:
            score, note = self.evaluate_single_prompt(prompt)
            scores.append(score)
            notes.append(note)
        return scores, notes

    def evaluate_single_prompt(self, prompt: str) -> tuple[float, dict]:
        '''
        Evaluates a single prompt

        Args:
            prompt: prompt to evaluate

        Returns:
            score: score of the prompt
            notes: notes
        '''

        generated_dialogues = [] # we will keep track of the dialogue generated
        feedbacks = [] # we will keep track of the feedback for each dialogue generated
        scores = [] # we will keep track of the scores for each dialogue generated
        
        for i in range(self.num_batch_runs):

            # sample a data point
            discussion_history, state_tuple, intended_actions, private_informations, roles, dialogue, speaking_order = self.data_loader.sample_data_point()
            state_info = self.data_loader.redact_state_info(state_tuple)

            # find out from roles which player is the role to evaluate
            player_to_eval = roles.index(self.role_to_evaluate)

            # get the first index at which the player is found in the speaking order
            if player_to_eval in speaking_order:
                role_index = speaking_order.index(player_to_eval)
                # truncate dialogue and speaking order
                dialogue = dialogue[:role_index]
                speaking_order = speaking_order[:role_index]

            # combine dialogue using prompt generator
            combined_dialogue = self.prompt_generator.gen_dialogue_description(dialogue, speaking_order)

            # create the dialogue generator
            dialogue_generator = DialogueGenerator(llm_model=self.llm_model, prompt_generator=self.prompt_generator, player=player_to_eval, private_information=private_informations[player_to_eval], tips=prompt)

            # get action intent
            action_intent = intended_actions[player_to_eval] # NOTE: allow option to use action planner? No, separate evaluator for that

            # generate the next dialogue
            next_dialogue = dialogue_generator.generate_dialogue(action_intent, combined_dialogue, *state_info)

            # --- now evaluate the next dialogue using the discriminator or search game simulation ---

            # create a dialogue discriminator for each other player that is a servant
            servant_dialogue_discriminators = [DialogueDiscriminator(llm_model=self.llm_model, prompt_generator=self.prompt_generator, player=player, private_information=private_informations[player], players=self.players, player_role=roles[player], known_sides=AvalonState.get_known_sides(player, roles)) for player in self.players if player != player_to_eval and roles[player] == 5]

            # create a dialogue discriminator for each other player that is evil
            evil_dialogue_discriminators = [DialogueDiscriminator(llm_model=self.llm_model, prompt_generator=self.prompt_generator, player=player, private_information=private_informations[player], players=self.players, player_role=roles[player], known_sides=AvalonState.get_known_sides(player, roles)) for player in self.players if player != player_to_eval and not AvalonBasicConfig.ROLES_TO_IS_GOOD[roles[player]]]

            if not AvalonBasicConfig.ROLES_TO_IS_GOOD[roles[self.role_to_evaluate]]:
                # evil player wants servants to think that they are good

                # use servant dialogue discriminators on next_dialogue, get_pgood_updates to get the response and the update for each servant
                feedback = []
                score = 0.0
                for servant in servant_dialogue_discriminators:
                    update, response = servant.get_pgood_updates(next_dialogue)
                    feedback.append(response)
                    score += update[player_to_eval][0]
                feedbacks.append(feedback)
                scores.append(score/len(servant_dialogue_discriminators))
            elif self.role_to_evaluate == 0: 
                # for merlin we want the evil players to not be able to identify us but for good players to be able to identify us
                feedback = []
                score = 0.0
                for evil_player in evil_dialogue_discriminators:
                    update, response = evil_player.get_pmerlin_updates(next_dialogue)
                    feedback.append(response)
                    score += 1-update[player_to_eval][0]
                feedbacks.append(feedback)
                scores.append(score/len(evil_dialogue_discriminators))
            else:
                raise ValueError("Not supported role to evaluate")
            
            generated_dialogues.append(next_dialogue)

        return float(np.mean(scores)), {"dialogues": generated_dialogues, "feedbacks": feedbacks}

