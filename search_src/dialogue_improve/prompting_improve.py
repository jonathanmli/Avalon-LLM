from search_src.searchlight.headers import Any
from search_src.searchlightimprove.headers import Evaluator
from search_src.searchlight.gameplay.simulators import GameSimulator
from search_src.searchlight.headers import *
from search_src.searchlight.gameplay.agents import SearchAgent
from search_src.searchlight.algorithms.mcts_search import SMMonteCarlo
from search_src.searchlight.datastructures.graphs import ValueGraph, PartialValueGraph
from search_src.searchlight.datastructures.adjusters import PUCTAdjuster
from search_src.searchlight.datastructures.estimators import UtilityEstimatorLast
from search_src.searchlight.classic_models import RandomRolloutValueHeuristic
from search_src.searchlightimprove.evaluators import SimulateSearchGameEvaluator
from search_src.searchlightimprove.llm_utils.llm_api_models import LLMModel

from .dialogue_discrimination import DialogueDiscriminator
from .dialogue_generator import DialogueGenerator
from .data_loader import DataLoader
from .prompt_generator import PromptGenerator
from ..Avalon.baseline_models_Avalon import AvalonState, AvalonBasicConfig, AvalonActionEnumerator, AvalonTransitor, AvalonActorEnumerator, AvalonGameEnvironment
from .action_planner import AvalonActionPlannerAgent





from typing import Type, Dict
import datetime




class PromptEvaluator(Evaluator):
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

    def set_num_batch_runs(self, num_batch_runs: int):
        '''
        Sets the number of batch runs

        Args:
            num_batch_runs: number of batch runs
        '''
        self.num_batch_runs = num_batch_runs

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

        # log self.batch_runs for debugging
        self.logger.info(f"Batch Runs: {self.num_batch_runs}")

        generated_dialogues = [] # we will keep track of the dialogue generated
        feedbacks = [] # we will keep track of the feedback for each dialogue generated
        scores = [] # we will keep track of the scores for each dialogue generated
        evaluated_players = [] # we will keep track of the players that have been evaluated
        generated_thoughts = [] # we will keep track of the thoughts generated
        
        for i in range(self.num_batch_runs):

            # sample a data point
            discussion_history, state_tuple, intended_actions, private_informations, roles, dialogue, speaking_order = self.data_loader.sample_data_point()
            state_info = self.data_loader.redact_state_info(state_tuple)

            # find out from roles which player is the role to evaluate
            player_to_eval = roles.index(self.role_to_evaluate)
            evaluated_players.append(player_to_eval)

            # get the first index at which the player is found in the speaking order
            if player_to_eval in speaking_order:
                role_index = speaking_order.index(player_to_eval)
                # truncate dialogue and speaking order
                dialogue = dialogue[:role_index]
                speaking_order = speaking_order[:role_index]

            # combine dialogue using prompt generator
            combined_dialogue = self.prompt_generator.gen_dialogue_description(dialogue, speaking_order)
            history_per_player = [self.prompt_generator.gen_summary_preamble(discussion_history[i]) for i in range(len(self.players))]

            # create the dialogue generator
            dialogue_generator = DialogueGenerator(llm_model=self.llm_model, prompt_generator=self.prompt_generator, player=player_to_eval, private_information=private_informations[player_to_eval], tips=prompt)

            # get action intent
            action_intent = intended_actions[player_to_eval] # NOTE: allow option to use action planner? No, separate evaluator for that

            # self.logger.info(f"Prompt: {prompt}, State: {state_info}, Action: {action_intent}")
            # print(f"Prompt: {prompt}, State: {state_info}, Action: {action_intent}")

            # generate the next dialogue
            next_dialogue, dialogue_info = dialogue_generator.generate_dialogue(action_intent, history_per_player[player_to_eval] + combined_dialogue, *state_info)
            generated_dialogues.append(next_dialogue)
            generated_thoughts.append(dialogue_info['thought'])

            # --- now evaluate the next dialogue using the discriminator or search game simulation ---

            # create a dialogue discriminator for each other player that is a servant
            servant_dialogue_discriminators = [DialogueDiscriminator(llm_model=self.llm_model, prompt_generator=self.prompt_generator, player=player, private_information=private_informations[player], players=self.players, player_role=roles[player], known_sides=AvalonState.get_known_sides(player, roles)) for player in self.players if player != player_to_eval and roles[player] == 5]

            # create a dialogue discriminator for each other player that is evil
            evil_dialogue_discriminators = [DialogueDiscriminator(llm_model=self.llm_model, prompt_generator=self.prompt_generator, player=player, private_information=private_informations[player], players=self.players, player_role=roles[player], known_sides=AvalonState.get_known_sides(player, roles)) for player in self.players if player != player_to_eval and not AvalonBasicConfig.ROLES_TO_IS_GOOD[roles[player]]]

            if not AvalonBasicConfig.ROLES_TO_IS_GOOD[self.role_to_evaluate]:
                # evil player wants servants to think that they are good

                # use servant dialogue discriminators on next_dialogue, get_pgood_updates to get the response and the update for each servant
                feedback = dict()
                score = 0.0
                for servant in servant_dialogue_discriminators:
                    # history = history_per_player[servant.player] + combined_dialogue + next_dialogue
                    history = history_per_player[servant.player] + combined_dialogue + next_dialogue
                    update, response = servant.get_pgood_updates(history)
                    feedback[servant.player]= (response, AvalonBasicConfig.ROLES[servant.player_role])
                    score += update[player_to_eval][0]
                feedbacks.append(feedback)
                scores.append(score/len(servant_dialogue_discriminators))
            elif self.role_to_evaluate == 0: 
                # for merlin we want the evil players to not be able to identify us but for good players to be able to identify us
                feedback = dict()
                evil_score = 0.0
                for evil_player in evil_dialogue_discriminators:
                    history = history_per_player[evil_player.player] + combined_dialogue + next_dialogue
                    update, response = evil_player.get_pmerlin_updates(history)
                    feedback[evil_player.player] = (response, AvalonBasicConfig.ROLES[evil_player.player_role])
                    evil_score -= update[player_to_eval][0]

                    # log the response for debugging
                    # self.logger.info(f"Evil Player: {evil_player.player}, Response: {response}")
                evil_score = evil_score/len(evil_dialogue_discriminators)
                good_score = 0.0
                for servant in servant_dialogue_discriminators:
                    # history = history_per_player[servant.player] + combined_dialogue + next_dialogue
                    history = history_per_player[servant.player] + combined_dialogue + next_dialogue
                    update, response = servant.get_pgood_updates(history)
                    feedback[servant.player]= (response, AvalonBasicConfig.ROLES[servant.player_role])
                    good_score += update[player_to_eval][0]
                good_score = good_score/len(servant_dialogue_discriminators)

                feedbacks.append(feedback)

                # log good and evil score
                self.logger.info(f"Evil Score: {evil_score}")
                self.logger.info(f"Good Score: {good_score}")
                
                # take minimum of the two scores
                score = min(evil_score, good_score)

                scores.append(score)
            else:
                raise ValueError("Not supported role to evaluate")
            
        return float(np.mean(scores)), {"dialogues": generated_dialogues, "feedbacks": feedbacks, "evaluated_players": evaluated_players, "thoughts": generated_thoughts, "scores": scores}

class PromptSSGEvaluator(SimulateSearchGameEvaluator):
    '''
    Evaluates the prompts using the discriminator
    '''

    def __init__(self, env: AvalonGameEnvironment, players: set, role_to_evaluate: int, data_loader: DataLoader, llm_model: LLMModel, prompt_generator: PromptGenerator, value_heuristic: ValueHeuristic2, num_scenarios: int = 1, num_total_game_sims: int = 10, rng: np.random.Generator = np.random.default_rng(), num_search_rollouts: int = 32):
        '''
        Args:
            env: game environment
            players: set of players
            role_to_evaluate: role to evaluate
            value_heuristic: value heuristic used by all players
            data_loader: data loader
            llm_model: llm model
            prompt_generator: prompt generator
            num_scenarios: number of evaluation runs (i.e. scenarios to sample from the data loader)
            num_game_sims: number of evaluation runs
            rng: random number generator
            against_benchmark: whether to evaluate against benchmark
            search_budget: search budget
            random_rollouts: number of random rollouts
        '''

        self.transitor = AvalonTransitor(env)
        self.actor_enumerator = AvalonActorEnumerator()
        self.action_enumerator = AvalonActionEnumerator(env)
        start_state = AvalonState.start_state_from_config(env.config)
        self.config = env.config

        game_simulator = GameSimulator(transitor=self.transitor, actor_enumerator=self.actor_enumerator, action_enumerator=self.action_enumerator, start_state=start_state)
        
        super().__init__(simulator=game_simulator, num_batch_runs=num_total_game_sims, players=players, rng=rng, stochastic_combinations=True)
        
        self.num_scenarios = num_scenarios
        self.role_to_evaluate = role_to_evaluate
        self.data_loader = data_loader
        self.llm_model = llm_model
        self.prompt_generator = prompt_generator
        self.value_heuristic = value_heuristic
        self.num_search_rollouts = num_search_rollouts


    def set_num_batch_runs(self, num_batch_runs: int):
        '''
        Sets the number of batch runs

        Args:
            num_batch_runs: number of batch runs
        '''
        self.num_batch_runs = num_batch_runs

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
        Evaluates a single prompt.
        The general idea sample scenarios from the data loader, feed them to the dialogue generator which uses the prompt to generate the next dialogue. Then feed the generated dialogue to the dialogue discriminator to get the feedback and beliefs. Next create a new AvalonActionPlannerAgent for each player and belief and simulate the game a couple of times to get the score.

        Args:
            prompt: prompt to evaluate

        Returns:
            score: score of the prompt
            notes: notes
        '''

        generated_dialogues = [] # we will keep track of the dialogue generated
        generated_thoughts = [] # we will keep track of the thoughts generated
        feedbacks = [] # we will keep track of the feedback for each dialogue generated
        scores = [] # we will keep track of the scores for each dialogue generated
        evaluated_players = [] # we will keep track of the players that have been evaluated
        search_estimates = [] # we will keep track of the search estimates
        
        for i in range(self.num_scenarios):

            # sample a data point
            discussion_history, state_tuple, intended_actions, private_informations, roles, dialogue, speaking_order = self.data_loader.sample_data_point()

            # log the data point for debugging
            
            # self.logger.info(f"State Tuple: {state_tuple}")
            # self.logger.info(f"Roles: {roles}")
            # self.logger.info(f"Intended Actions: {intended_actions}")
            # self.logger.info(f"Speaking Order: {speaking_order}")
            # self.logger.info(f"Dialogue: {dialogue}")
            # self.logger.info(f"Discussion History: {discussion_history}") # compromised
            # self.logger.info(f"Private Informations: {private_informations}") # compromised
            
            

            # get the roles in the state tuple
            state_tuple_roles = state_tuple[-1]

            # assert that the roles in the state tuple are the same as the roles in the data point
            # assert roles == state_tuple_roles, f"Roles in state tuple: {state_tuple_roles} do not match roles in data point: {roles}"
            
            # NOTE: hack to fix henry's poor data generation
            roles = state_tuple_roles # FIXME: temporary hack
            discussion_history = ["" for _ in range(len(roles))] # FIXME: temporary hack
            private_informations = [] # FIXME: temporary hack
            for i in range(len(roles)):
                role_id = roles[i]
                role_str = AvalonBasicConfig.ROLES[role_id]
                known_sides = AvalonState.get_known_sides(i, roles)
                # p_info = f"You are Player {i}, with identity {role_str}. You are on the side of {'Good' if AvalonBasicConfig.ROLES_TO_IS_GOOD[role_id] else 'Evil'}. You know that the sides of the other players are {known_sides} where 1 (True) is Good, 0 (False) is Evil, and -1 is unknown whether they are Good or Evil. Please do not forget your identity throughout the game."
                p_info = f"You are Player {i}, with identity {role_str}. You are on the side of {'Good' if AvalonBasicConfig.ROLES_TO_IS_GOOD[role_id] else 'Evil'}. You know that players {[j for j in range(len(roles)) if known_sides[j] == 1]} are Good and players {[j for j in range(len(roles)) if known_sides[j] == 0]} are Evil. The rest you do not know. Please do not forget your identity throughout the game."
                private_informations.append(p_info)

            # self.logger.info(f"State Tuple: {state_tuple}")
            # self.logger.info(f"Roles: {roles}")
            # self.logger.info(f"Intended Actions: {intended_actions}")
            # self.logger.info(f"Speaking Order: {speaking_order}")
            # # self.logger.info(f"Dialogue: {dialogue}")
            # self.logger.info(f"Discussion History: {discussion_history}") # compromised
            # self.logger.info(f"Private Informations: {private_informations}") # compromised

            state_info = self.data_loader.redact_state_info(state_tuple)

            # find out from roles which player is the role to evaluate
            player_to_eval = roles.index(self.role_to_evaluate)
            evaluated_players.append(player_to_eval)

            # get the first index at which the player is found in the speaking order
            if player_to_eval in speaking_order:
                role_index = speaking_order.index(player_to_eval)
                # truncate dialogue and speaking order
                dialogue = dialogue[:role_index]
                speaking_order = speaking_order[:role_index]

            # combine dialogue using prompt generator
            combined_dialogue = self.prompt_generator.gen_dialogue_description(dialogue, speaking_order)
            history_per_player = [self.prompt_generator.gen_summary_preamble(discussion_history[i]) for i in range(len(self.players))]

            # create the dialogue generator
            dialogue_generator = DialogueGenerator(llm_model=self.llm_model, prompt_generator=self.prompt_generator, player=player_to_eval, private_information=private_informations[player_to_eval], tips=prompt)

            # get action intent
            action_intent = intended_actions[player_to_eval] # NOTE: allow option to use action planner? No, separate evaluator for that

            # self.logger.info(f"Prompt: {prompt}, State: {state_info}, Action: {action_intent}")
            # print(f"Prompt: {prompt}, State: {state_info}, Action: {action_intent}")

            # generate the next dialogue
            next_dialogue, dialogue_info = dialogue_generator.generate_dialogue(action_intent, history_per_player[player_to_eval] + combined_dialogue, *state_info)
            generated_dialogues.append(next_dialogue)
            generated_thoughts.append(dialogue_info['thought'])

            # --- now create a new action planner agent for each player and simulate the game ---
            agents: dict[Any, AvalonActionPlannerAgent] = dict()
            player_notes = dict()
            feedback = dict()
            for player in self.players:
                private_info = private_informations[player]
                role = roles[player]
                known_sides = AvalonState.get_known_sides(player, roles)
                new_agent = AvalonActionPlannerAgent(config=self.config, player=player, private_information=private_info, player_role=role, known_sides=known_sides, value_heuristic=self.value_heuristic, llm_model=self.llm_model, num_rollouts=self.num_search_rollouts, rng=self.rng)
                agents[player] = new_agent

                # have the agent observe the dialogue
                history = history_per_player[player] + combined_dialogue + next_dialogue
                # player_notes = new_agent.observe_dialogue(history)

                # combined_response = player_notes['pgood_response'] + player_notes['pmerlin_response']
                combined_response = ''
                feedback[player] = (combined_response, AvalonBasicConfig.ROLES[role])

            # we need to set the start state of our simulator to the current state
            start_state = AvalonState.init_from_state_tuple(self.config, *state_tuple[1:])

            # print start state for debugging
            self.logger.info(f"Start State: {start_state}")
            self.logger.info(f"Agents: {agents}")

            starttime = datetime.datetime.now()
            # simulate the game
            avg_game_scores, trajectories = self.simulator.simulate_games(agents=agents, num_games=self.num_batch_runs, start_state=start_state, display=True)

            # get search estimate of player_agent
            player_agent = agents[player_to_eval]
            start_node = player_agent.graph.get_node(start_state)
            if start_node is None:
                # search_estimate = 0
                # create the node in the graph
                self.logger.info(f"Creating Start Node")
                player_agent.graph.add_state(start_state)
                self.logger.info(f"Start Node is None")
            else:
                search_estimate = player_agent.graph.get_estimated_value(start_node, player_to_eval)

            # log how long it took to simulate the game
            endtime = datetime.datetime.now()
            elapsed_time = endtime - starttime
            self.logger.info(f"Simulation took {elapsed_time}")

            # log average game scores for the player to evaluate
            self.logger.info(f"Average Game Scores: {avg_game_scores[player_to_eval]}")

            # log search estimate for the player to evaluate
            self.logger.info(f"Search Estimate: {search_estimate}")

            # log role of player to evaluate
            self.logger.info(f"Role of Player to Evaluate: {AvalonBasicConfig.ROLES[self.role_to_evaluate]}")


            # append the score for the evaluated player
            scores.append(avg_game_scores[player_to_eval])
            feedbacks.append(feedback)

            # get the agent corresponding to the player to evaluate
            player_agent = agents[player_to_eval]
            start_node = player_agent.graph.get_node(start_state)
            if start_node is None:
                search_estimate = 0
            else:
                search_estimate = player_agent.graph.get_estimated_value(start_node, player_to_eval)
            search_estimates.append(search_estimate)

            
            


        return float(np.mean(scores)), {"dialogues": generated_dialogues, "feedbacks": feedbacks, "evaluated_players": evaluated_players, "search_estimates": search_estimates, "thoughts": generated_thoughts}