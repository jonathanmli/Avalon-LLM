from search_src.searchlight.headers import *
from pydantic import BaseModel
from typing import ClassVar, List, Optional, Dict
from search_src.Avalon.engine import AvalonBasicConfig, AvalonGameEnvironment
import itertools
import copy
import re
from search_src.searchlightimprove.llm_utils.llm_api_models import LLMModel
from search_src.searchlightimprove.prompts.prompt_generators import PromptGenerator
from .prompts import GAME_RULES, HEURISTICS_FUNCTION_SIGNATURE


'''
NOTE: Avalon has multiplayers, not two players like GOPS
NOTE: Some bugs in MCTS search may appear 
'''


class AvalonState(HiddenState):

    def __init__(self, config: AvalonBasicConfig, quest_leader: int, phase: int, turn: int, round: int, done: bool, good_victory: bool, quest_team: frozenset[int], team_votes: tuple[bool, ...], quest_votes: tuple[bool, ...], quest_results: tuple[bool, ...], roles: tuple[int, ...],):
        '''
        Game states is a dictionary mapping from relevant game stats to their values for Avalon
        Should probably include:
        - Number of players
        - Number of good players
        - Number of quests
        - Number of fails
        - Current leader
        - Current team
        - Current quest
        - Current vote
        - Current mission
        - Current mission vote
        - Current mission result
        - Current phase
        etc... 
        '''
        self.config = config
        self.quest_leader = quest_leader
        self.phase = phase
        self.turn = turn
        self.round = round
        self.done = done
        self.good_victory = good_victory
        self.quest_team = quest_team
        self.team_votes = team_votes
        self.quest_votes = quest_votes
        self.quest_results = quest_results

        # the following information is hidden
        self.roles = roles

        # make sure all elements in here are basic types or tuple, frozenset to make it hashable and easily comparable
        id = tuple([self.quest_leader, self.phase, self.turn, self.round, self.done,
                    self.quest_team, self.team_votes, self.quest_votes,
                    self.quest_results, self.roles])
        
        self.player_to_information_set = dict()
        for i in range(self.config.num_players):
            known_sides, self_role  = self.get_private_information(i)
            information_set =  tuple([self.quest_leader, self.phase, self.turn, self.round, self.done, self.quest_team, self.team_votes, self.quest_results, known_sides, self_role])
            self.player_to_information_set[i] = information_set
        super().__init__(id)

    def get_state_tuple(self):
        roles = tuple([int(role) for role in self.roles])
        return (self.config.num_players, self.quest_leader, self.phase, self.turn, self.round, self.done, list(self.quest_team), self.team_votes, self.quest_votes, self.quest_results, roles)

    def get_private_information(self, player: int) -> tuple[tuple[int, ...], int]:
        '''
        Returns the private information of the actor
        '''

        is_good = self.get_is_good()
        # if player is Merlin or evil, return all sides
        if self.roles[player] == 0 or not is_good[player]:
            return tuple(is_good), self.roles[player]
        # otherwise return list of -1 for unknown
        else:
            return tuple([-1 if i != player else 1 for i in range(self.config.num_players)]), self.roles[player]
        
    def get_information_set(self, actor):
        return self.player_to_information_set[actor]
    
    @staticmethod
    def get_known_sides(player: int, roles) -> tuple[int, ...]:
        '''
        Returns the private information of the actor
        '''
        is_good = [AvalonBasicConfig.ROLES_TO_IS_GOOD[role] for role in roles]
        # if player is Merlin or evil, return all sides
        if roles[player] == 0 or not is_good[player]:
            return tuple(is_good)
        # otherwise return list of -1 for unknown
        else:
            return tuple([-1 if i != player else 1 for i in range(len(roles))])
        

    @staticmethod
    def init_from_env(env: AvalonGameEnvironment):
        '''
        Copy values from the environment to the state
        '''
        # self.num_players = env.config.num_players
        # self.num_good = env.config.num_good
        # self.num_evil = env.config.num_evil
        # self.num_players_for_quest = env.config.num_players_for_quest
        # self.num_fails_for_quest = env.config.num_fails_for_quest
        config = env.config

        quest_leader = env.quest_leader
        phase = env.phase
        turn = env.turn
        round = env.round
        done = env.done
        # self.good_victory = env.good_victory 

        # make the following tuples so that they are immutable
        quest_team = frozenset(env.quest_team)
        team_votes = tuple(env.team_votes)
        quest_votes = tuple(env.quest_votes)
        quest_results = tuple(env.quest_results)
        roles = tuple(env.roles)


        return AvalonState(config, quest_leader, phase, turn, round, done, env.good_victory, quest_team, team_votes, quest_votes, quest_results, roles)

    def copy(self):
        '''
        Returns a copy of the state

        We want to keep the env the same, but copy everything else
        '''
        return AvalonState(self.config, self.quest_leader, self.phase, self.turn, self.round, self.done, self.good_victory, self.quest_team, self.team_votes, self.quest_votes, self.quest_results, self.roles)

    def get_assassin(self):
        '''
        Returns the assassin
        '''
        for i, role in enumerate(self.roles):
            if role == 7:
                return i
        
        raise ValueError('No assassin found')
    
    def initial_state_randomize(self, rng: np.random.Generator = np.random.default_rng()):
        '''
        Randomizes the initial state
        '''
        # only do this if the phase is 0 and the round is 0 and the turn is 0
        # if self.phase == 0 and self.round == 0 and self.turn == 0:
            # randomly permute the roles
        self.roles = tuple(rng.permutation(self.roles))
        # print('roles = ', self.roles)

    def get_is_good(self):
        '''
        Returns the side of each player according to the role (True for good, False for evil)
        '''
        is_good = [self.config.ROLES_TO_IS_GOOD[role] for role in self.roles]
        return is_good
    
    def get_roles_in_str_list(self):
        '''
        Returns the roles in string format
        '''
        return [self.config.ROLES[role] for role in self.roles]

class AvalonActorEnumerator(ActorEnumerator):

    def __init__(self):
        super().__init__()

    def _enumerate(self, state: AvalonState) -> set:
        '''
        Enumerates the actors for the given state

        Args:
            state: current state

        Returns:
            actors: set of actors
        '''

        if state.phase == 0:
            actors = set([state.quest_leader])

        if state.phase == 1:
            actors = set(range(state.config.num_players))

        if state.phase == 2:
            actors = set(state.quest_team)

        if state.phase == 3:
            actors = set([state.get_assassin()])

        if state.done:
            actors = set()

        return actors


class AvalonActionEnumerator(ActionEnumerator):

    def __init__(self, avalon_env: AvalonGameEnvironment):
        super().__init__()
        self.num_players_per_quest = avalon_env.config.num_players_for_quest
        num_players = avalon_env.config.num_players
        self.all_players = set(range(num_players))
        
        self.player_combinations = dict()
        for quest_size in set(avalon_env.config.num_players_for_quest):
            # get all combinations of players of size quest_size
            combinations = list(itertools.combinations(self.all_players, quest_size))
            self.player_combinations[quest_size] = set([frozenset(combine) for combine in combinations])


    def _enumerate(self, state: AvalonState, actor) -> set:
        '''
        Enumerates the actions for the given state and actor

        Args:
            state: current state
            actor: actor to enumerate actions for

        Returns:
            actions: list of actions
        '''

        if state.phase == 0:
            turn = state.turn
            actions = self.player_combinations[self.num_players_per_quest[turn]]

        if state.phase == 1:
            actions = {0, 1}

        if state.phase == 2:
            actions = {0, 1}

        if state.phase == 3:
            actions = self.all_players

        return actions


class AvalonTransitor(ForwardTransitor2):

    def __init__(self, env: AvalonGameEnvironment):
        self.env = env
        super().__init__()


    def _transition(self, state: AvalonState, actions: dict) -> Tuple[AvalonState, dict[Any, float], dict]:
        '''
        Transits to the next state given the current state and action

        Args:
            state: current state
            actions: actions taken by the actors

        Returns:
            next_state: next state, reward, notes
        '''

        # first extract all relevant information from the state to the environment engine
        self.env.config = state.config
        self.env.quest_leader = state.quest_leader
        self.env.phase = state.phase
        self.env.turn = state.turn
        self.env.round = state.round
        self.env.done = state.done
        self.env.good_victory = state.good_victory
        self.env.quest_team = state.quest_team
        self.env.team_votes = list(state.team_votes)
        self.env.quest_votes = list(state.quest_votes)
        self.env.quest_results = list(state.quest_results)
        self.env.roles = np.array(state.roles)
        self.env.is_good = np.array(state.get_is_good())
        # state.assassin


        if self.env.phase == 0:
            for key in actions:
                action = actions[key]
            self.env.choose_quest_team(action, self.env.quest_leader)
            # self.env.phase, self.env.done, self.env.quest_leader = 

        elif self.env.phase == 1:
            action = []
            for key in actions:
                action.append(actions[key])
            self.env.gather_team_votes(action)

            # self.env.phase, self.env.done, team_votes_result = 


        elif self.env.phase == 2:
            action = []
            for key in actions:
                action.append(actions[key])
            #print(AvalonState(self.env))
            self.env.gather_quest_votes(action)
            # self.env.phase, self.env.done, quest_vote_result, num_fails = 


        elif self.env.phase == 3:
            player = state.get_assassin()
            for key in actions:
                action = actions[key]

            self.env.choose_assassination_target(player, action)

            # self.env.phase, self.env.done, self.env.good_victory = 
        
        # now extract all relevant information from the environment engine to the next state
        next_state = AvalonState.init_from_env(self.env)

        reward = dict()
        for i in range(self.env.config.num_players):
            reward[i] = 0

        if self.env.done and self.env.good_victory:
            for ii in range(self.env.config.num_players):
                (role_ii, role_ii_name, role_ii_good) = self.env.get_role(ii)
                if role_ii_good == True:
                    reward[ii] = 1
                else:
                    reward[ii] = -1

        elif self.env.done and not self.env.good_victory:
            for ii in range(self.env.config.num_players):
                (role_ii, role_ii_name, role_ii_good) = self.env.get_role(ii)
                if role_ii_good == True:
                    reward[ii] = -1
                else:
                    reward[ii] = 1

        notes = {}
        return (next_state, reward, notes)

class AvalonLLMFunctionalValueHeuristic(ValueHeuristic2):
    '''
    Functional value heuristic for LLMs
    '''

    EVAL_TEST_STATES = [
        {'quest_leader': 4, 'phase': 0, 'turn': 0, 'round': 1, 'quest_team': frozenset({3, 4}), 'historical_team_votes': (1, 0, 1, 0, 0), 'historical_quest_results': (), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Servant', 'Merlin', 'Servant', 'Assassin', 'Minion'], 'is_good': [True, True, True, False, False]},
    ]

    def __init__(self, func, parse_first = False):
        '''
        Args:
            model: LLM model
        '''
        super().__init__()
        if parse_first:
            func = AvalonLLMFunctionalValueHeuristic.parse_llm_generated_function(func)
        self.attach_function(func)

    def attach_function(self, function_str: str):
        '''
        Attach a function to the instance
        '''
        # Execute the function definition within the local scope of __init__
        exec(function_str, globals(), locals())
        
        # Attach the dynamically defined function to the instance
        self._llm_evaluate = locals()['evaluate_state']

    @staticmethod
    def generate_seed_function(model: LLMModel, prompt_gen: PromptGenerator) -> str:
        '''
        Generates a seed function from the model
        '''
        prompt = prompt_gen.gen_seed_thought_prompt()
        thought = model.generate(prompt)
        prompt = prompt_gen.gen_seed_function_prompt(thought)
        function_str = model.generate(prompt)
        return function_str[0]
    
    @staticmethod
    def parse_llm_generated_function(function_str: str, safe_mode: bool=False) -> str:
        '''
        Parses the function generated by the LLM
        '''
        pattern = re.compile(r'(?ms)^def\s+evaluate_state\s*\(.*?\)\s*.*?:\s*.*?(?=^\w|\Z)', re.MULTILINE)
        match = re.search(pattern, function_str)
        if match:
            # print(f"Parsed Function:\n{match.group()}")
            # logging.info(match.group()) #TODO: fix logging to match the new logging framework
            parsed_func = match.group()
            function_lines = parsed_func.split('\n')
            last_line_id = -2
            for line_id in range(len(function_lines)-1, -1, -1):
                if function_lines[line_id].lstrip().startswith("return"):
                    last_line_id = line_id
                    break
            return '\n'.join(function_lines[:last_line_id+1])
        else:
            # print("Function not found!!!!!!!!")
            if not safe_mode:
                raise ValueError("Parsing error: Function not found")
            return None

    @staticmethod
    def convert_state_to_input(state: AvalonState) -> dict[str, Any]:
        '''
        Converts the state to the input format for the model
        '''
        input_dict = dict()
        input_dict['quest_leader'] = state.quest_leader
        input_dict['phase'] = state.phase
        input_dict['turn'] = state.turn
        input_dict['round'] = state.round
        input_dict['quest_team'] = state.quest_team
        input_dict['historical_team_votes'] = state.team_votes
        input_dict['historical_quest_results'] = state.quest_results
        # input_dict['roles'] = state.roles
        input_dict['players'] = set(range(state.config.num_players))


        input_dict['num_good'] = state.config.num_good
        input_dict['num_participants_per_quest'] = state.config.num_players_for_quest
        input_dict['num_fails_per_quest'] = state.config.num_fails_for_quest



        input_dict['roles'] = state.get_roles_in_str_list()
        input_dict['is_good'] = state.get_is_good()


        return input_dict

    @staticmethod
    def test_evaluate_static(function_str, safe_mode = False) -> bool:
        '''
        Test the evaluate function
        '''
        if safe_mode:
            # Test the evaluate function
            try:
                # Execute the function definition within the local scope of __init__
                exec(function_str, globals(), locals())
                
                # Attach the dynamically defined function to the instance
                llm_evaluate = locals()['evaluate_state']

                # print('successfully defined function')

                for state in AvalonLLMFunctionalValueHeuristic.EVAL_TEST_STATES:
                    player_to_score, notes = llm_evaluate(state)
                    # assert that player_to_score is a dictionary
                    assert isinstance(player_to_score, dict)
                    # assert that all values in player_to_score are numbers
                    assert all(isinstance(value, (int, float)) for value in player_to_score.values())
                    # assert that notes is a dictionary
                    assert isinstance(notes, dict)

                # print('successfully passed test')
            except Exception as e:  # Capture the exception as 'e'
                print(f"An exception occurred: {e}")  # Print the exception for debugging
                return False
            return True
        else:
            # Execute the function definition within the local scope of __init__
            exec(function_str, globals(), locals())
            
            # Attach the dynamically defined function to the instance
            llm_evaluate = locals()['evaluate_state']

            for state in AvalonLLMFunctionalValueHeuristic.EVAL_TEST_STATES:
                (player_value, opponent_value), notes = llm_evaluate(state)
                # assert that both values are numbers
                assert isinstance(player_value, (int, float))
                assert isinstance(opponent_value, (int, float))
                # assert that notes is a dictionary
                assert isinstance(notes, dict)

            return True

    def _evaluate(self, state: AvalonState) -> tuple[dict, dict]:
        # Prepare input
        input_dict = AvalonLLMFunctionalValueHeuristic.convert_state_to_input(state)

        # use the function to calculate the value
        try:
            winrates, notes = self._llm_evaluate(input_dict)
            # assert that winrates is a dictionary
            assert isinstance(winrates, dict)
            # assert that winrates.keys() == input_dict['players']
            assert set(winrates.keys()) == input_dict['players']
        # raise an error if the function is not defined properly
        except Exception as e:
            logging.warning(f"state dictionary: {str(input_dict)}")
            # print('state tuple', state_tup)
            # NOTE: add any printed tuples to EVAL_TEST_STATES for future testing
            raise ValueError(f"Function not defined properly: {e}")
        
        # convert winrates to zero sum scores (times 2 minus 1)
        winrates = {player: 2*winrate - 1 for player, winrate in winrates.items()}
        return winrates, notes