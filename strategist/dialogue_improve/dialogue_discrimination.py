from strategist.searchlight.utils import AbstractLogged
from strategist.searchlightimprove.llm_utils.llm_api_models import LLMModel
from .prompt_generator import PromptGenerator

import numpy as np
import re
import ast

class DialogueDiscriminator(AbstractLogged):
    '''
    Class for dialogue discrimination. Dialogue discrimination is the process of converting the dialogue from the previous round and summary of discussions from rounds before that into a better representation (i.e. numerical beliefs). The dialogue discriminator is used to update the beliefs of the agent based on the input string.
    '''
    NUM_TRIES = 8

    def __init__(self, llm_model: LLMModel, prompt_generator: PromptGenerator, player: int, players: set[int], update_multipliers = {2: 1, 1: 0.5, 0: 1, -1: -0.5, -2: -1}):
        '''
        
        Args:
            llm_model: LLM model for dialogue discrimination
            prompt_generator: prompt generator for generating prompts
            known_sides: list of sides that are known to the agent
            player: player index of the agent
            player_role: role of the player
            private_information: the private information string given to the player
        
        '''
        super().__init__()
        self.llm_model = llm_model # LLM model for dialogue discrimination
        self.prompt_generator = prompt_generator
        self.update_multipliers = update_multipliers
        self.player = player
        self.players = players
        self.player_role = None
        self.logits_is_good = np.zeros(len(players))
        self.logits_is_merlin = np.zeros(len(players))
        # self.reset(known_sides, player_role, )

    def reset(self, known_sides: tuple[int, ...], player_role: int, ):
        # we will keep track of logits for each player being good and merlin instead of the probabilities themselves
        self.logits_is_good = np.zeros(len(known_sides))
        self.logits_is_merlin = np.zeros(len(known_sides))

        if player_role == 5: # servant
            # set p_is_good 0.5 for all players that are known to be good other than self
            # we just need to set logit for player to inf
            self.logits_is_good[self.player] = np.inf
        else:
            # assert that -1 is not in known_sides
            assert -1 not in known_sides
            # set self.logits_is_good to inf if player is known to be good (known_side == 1) and -inf if player is known to be evil (known_side == 0) in known_sides
            self.logits_is_good[known_sides == 1] = np.inf
            self.logits_is_good[known_sides == 0] = -np.inf
            # self.logger.info(f"Player: {player}, logit: {self.logits_is_good}")
            

        if player_role == 0:
            self.logits_is_merlin[self.player] = np.inf
        else:
            # set self.logits_is_merlin to -inf if player is known to be evil (known_side == 0) in known_sides. otherwise set it to 0
            self.logits_is_merlin[known_sides == 0] = -np.inf
            # self.logger.info(f"Player: {player}, logit: {self.logits_is_merlin}")
            
        
        self.player_role = player_role
        
        
    def check_role_equivalent(self, role: int) -> bool:
        '''
        Check if the role is equivalent to the player role

        Args:
            role: role to check

        Returns:
            True if the role is equivalent to the player role, False otherwise
        '''
        return role == self.player_role
        
    
    def update_beliefs(self, history: str) -> dict:
        '''
        Updates the beliefs based on the input string. The input string might include the dialogue from the previous round and summary of discussions from rounds before that.

        Args:
            history: input string to update beliefs

        Returns:
            notes: dictionary containing the updates and responses
        '''
        if self.player_role == 5: # servant
            # for servant we need to update both p_is_good and p_is_merlin
            pgood_updates, pgood_response = self.get_pgood_updates(history)
            pmerlin_updates, pmerlin_response = self.get_pmerlin_updates(history)
            for player in self.players:
                self.logits_is_good[player] += pgood_updates.get(player, (0, ''))[0]
            for player in self.players:
                # self.logger.info(f"Player: {player}, update: {update}")
                # self.logger.info(f"Player: {player}, logit: {self.logits_is_merlin}")
                self.logits_is_merlin[player] += pmerlin_updates.get(player, (0, ''))[0]
        elif self.player_role == 0: # merlin
            # for merlin no updates are needed
            pmerlin_response = ''
            pmerlin_updates = {}
            pgood_updates = {}
            pgood_response = ''

        elif self.player_role == 6 or 7:
            # for Minion and Assassin we only need to update p_is_merlin
            pmerlin_updates, pmerlin_response = self.get_pmerlin_updates(history)
            for player, (update, _) in pmerlin_updates.items():
                self.logits_is_merlin[player] += update
            pgood_updates = {}
            pgood_response = ''
        else:
            raise ValueError("Not supported player role")
        
        return {'pgood_updates': pgood_updates, 'pmerlin_updates': pmerlin_updates, 'pgood_response': pgood_response, 'pmerlin_response': pmerlin_response}
        
    
    def get_pgood_updates(self, history: str) -> tuple[dict[int, tuple[int, str]], str]:
        '''
        Returns the updates to the p_is_good beliefs based on the input string

        Args:
            history: input string to update beliefs
            response: response from the LLM model


        NOTE: should we also include the state of the game in the input string?
        '''
        # discriminate using the LLM model to get the p_is_good updates
        # TODO: incorporate context here
        pgood_prompt = self.prompt_generator.generate_pgood_belief_discrimination_prompt(history, self.players)

        for _ in range(self.NUM_TRIES):
            response = self.llm_model.generate(pgood_prompt, 1)[0]
            # print("good: \n", response)
            try:
                return self.extract_and_parse_dictionary(response), response
            except ValueError as e:
                self.logger.warning(f"Error parsing the dictionary string: {str(e)}")
                pgood_prompt = self.prompt_generator.generate_parsing_error_prompt(pgood_prompt, response, str(e))
        raise ValueError(f"Failed to parse the dictionary string after {self.NUM_TRIES} attempts.")
    
    def get_pmerlin_updates(self, history: str) -> tuple[dict[int, tuple[int, str]], str]:
        '''
        Returns the updates to the p_is_merlin beliefs based on the input string

        Args:
            history: input string to update beliefs
            response: response from the LLM model
        '''
        # discriminate using the LLM model to get the p_is_merlin updates
        pmerlin_prompt = self.prompt_generator.generate_pmerlin_belief_discrimination_prompt(history, self.players)

        for _ in range(self.NUM_TRIES):
            response = self.llm_model.generate(pmerlin_prompt, 1)[0]
            # print("merlin: \n", response)
            try:
                return self.extract_and_parse_dictionary(response), response
            except ValueError as e:
                self.logger.warning(f"Error parsing the dictionary string: {str(e)}")
                pmerlin_prompt = self.prompt_generator.generate_parsing_error_prompt(pmerlin_prompt, response, str(e))
        raise ValueError(f"Failed to parse the dictionary string after {self.NUM_TRIES} attempts.")
    
    @staticmethod
    def extract_and_parse_dictionary(update: str) -> dict[int, tuple[int, str]]:
        '''
        Extracts and parses a dictionary from a block of descriptive text. This version corrects regex patterns.

        Args:
            update: A string that includes descriptive text followed by a dictionary.

        Returns:
            A dictionary mapping each player index to a tuple containing an integer and a string
            that describes the change in belief about the likelihood of the player being Evil.
        '''
        try:
            # Corrected regex pattern to handle optional spaces, and numeric ranges appropriately
            regex_pattern = r"\{\s*\d+\s*:\s*\(-?[0-2],\s*'[a-zA-Z\s]+?'\)\s*(,\s*\d+\s*:\s*\(-?[0-2],\s*'[a-zA-Z\s]+?'\)\s*)*\}"
            match = re.search(regex_pattern, update)
            if match:
                dict_str = match.group(0)
                result = ast.literal_eval(dict_str)
                # Ensure that the result is a dictionary and the values are tuples as expected
                if isinstance(result, dict) and all(isinstance(v, tuple) and len(v) == 2 for v in result.values()):
                    return result
                else:
                    raise ValueError("The extracted data does not match the expected format (dict of int to tuples).")
            else:
                raise ValueError("No dictionary-like string found in the text.")
        except (SyntaxError, ValueError) as e:
            raise ValueError("Error parsing the dictionary string: " + str(e))

    def get_p_is_good(self) -> np.ndarray:
        '''
        Returns the p_is_good beliefs
        '''
        # convert logits_is_good to probabilities. recall that if logit == -inf, then probability is 0 and if logit == inf, then probability is 1
        return 1 / (1 + np.exp(-self.logits_is_good))
    
    def get_p_is_merlin(self) -> np.ndarray:
        '''
        Returns the p_is_merlin beliefs

        TODO: check that this is correct
        '''
        # convert logits_is_merlin to probabilities. recall that if logit == -inf, then probability is 0 and if logit == inf, then probability is 1
        # we must also normalize the probabilities so that they sum to 1
        p_is_merlin = 1 / (1 + np.exp(-self.logits_is_merlin))
        return p_is_merlin / np.sum(p_is_merlin)

    def get_beliefs(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Returns the beliefs of the agent
        '''
        return self.get_p_is_good(), self.get_p_is_merlin()
        
        