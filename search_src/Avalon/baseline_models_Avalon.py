from collections.abc import Hashable
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
from numpy.typing import NDArray


'''
NOTE: Avalon has multiplayers, not two players like GOPS
NOTE: Some bugs in MCTS search may appear 
'''


class AvalonState(StateTemplate):

    simultaneous_actions: tuple[tuple[Any, Any], ...]

    def __init__(self, config: AvalonBasicConfig, quest_leader: int, phase: int, turn: int, round: int, done: bool, good_victory: bool, quest_team: tuple[int, ...], team_votes: tuple[bool, ...], quest_votes: tuple[bool, ...], quest_results: tuple[bool, ...], roles: tuple[int, ...], acting_players: tuple[int, ...], simultaneous_actions: tuple):
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
        self.acting_players = acting_players # players that still need to act, in order
        self.simultaneous_actions = simultaneous_actions # tuples of (player, action)

        # make sure all elements in here are basic types or tuple, frozenset to make it hashable and easily comparable
        id = tuple([self.quest_leader, self.phase, self.turn, self.round, self.done,
                    self.quest_team, self.team_votes, self.quest_votes,
                    self.quest_results, self.roles, self.acting_players, self.simultaneous_actions])
        super().__init__(id)

    def get_state_tuple(self):
        return (self.config.num_players, self.quest_leader, self.phase, self.turn, self.round, self.done, self.good_victory, self.quest_team, self.team_votes, self.quest_votes, self.quest_results, self.roles)

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

    def get_private_information_string(self, player: int) -> str:
        '''
        Returns the private information in a paragraph format
        '''
        known_sides, self_role = self.get_private_information(player)
        known_sides_str = ', '.join([f'Player {i} is {"Good" if side else "Evil"}' for i, side in enumerate(known_sides) if side != -1])
        out = f"""Your role is {self.config.ROLES[self_role]} and you are on the side of {"Good" if self.config.ROLES_TO_IS_GOOD[self_role] else "Evil"}.
        
        You know the following information:
        {known_sides_str}
        """
        return out
    
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
    def start_state_from_config(config: AvalonBasicConfig):
        '''
        Returns the start state from the config
        '''
        return AvalonState(config, 0, -1, 0, 0, False, False, tuple(), tuple(), tuple(), tuple(), tuple([0 for i in range(config.num_players)]), tuple([-1]), tuple())
        

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
        quest_team = tuple(sorted(list(env.quest_team)))
        team_votes = tuple(env.team_votes)
        quest_votes = tuple(env.quest_votes)
        quest_results = tuple(env.quest_results)
        roles = tuple(env.roles)

        if phase == 0:
            acting_players = tuple([quest_leader])
        elif phase == 1:
            acting_players = tuple(range(config.num_players))
        elif phase == 2:
            acting_players = tuple(env.quest_team)
        elif phase == 3:
            acting_players = tuple([env.get_assassin()])

        # if we init from env, we assume nobody has taken any simultaneous actions yet
        return AvalonState(config, quest_leader, phase, turn, round, done, env.good_victory, quest_team, team_votes, quest_votes, quest_results, roles, acting_players, tuple())
    
    @staticmethod
    def init_from_state_tuple(config: AvalonBasicConfig, quest_leader: int, phase: int, turn: int, round: int, done: bool, good_victory: bool, quest_team: tuple[int, ...], team_votes: tuple[bool, ...], quest_votes: tuple[bool, ...], quest_results: tuple[bool, ...], roles: tuple[int, ...], ):
        if phase == -1:
            acting_players = tuple([-1])
        elif phase == 0:
            acting_players = tuple([quest_leader])
        elif phase == 1:
            acting_players = tuple(range(config.num_players))
        elif phase == 2:
            acting_players = tuple(quest_team)
        elif phase == 3:
            # find assassin by looking through roles
            assassin = 0
            for i, role in enumerate(roles):
                if role == 7:
                    assassin = i
                    break
            acting_players = tuple([assassin])
        return AvalonState(config, quest_leader, phase, turn, round, done, good_victory, quest_team, team_votes, quest_votes, quest_results, roles, acting_players, tuple())


    def copy(self):
        '''
        Returns a copy of the state

        We want to keep the env the same, but copy everything else
        '''
        return AvalonState(self.config, self.quest_leader, self.phase, self.turn, self.round, self.done, self.good_victory, self.quest_team, self.team_votes, self.quest_votes, self.quest_results, self.roles, self.acting_players, self.simultaneous_actions)

    def get_acting_player(self):
        '''
        Returns the acting player
        '''
        return self.acting_players[0]
    
    def next_simulaneous_state_copy(self, action: Any):
        '''
        Returns a copy of the state

        Args:
            action: the action taken by the acting player
        '''
        # remove the acting player from the acting players
        acting_players = self.acting_players[1:]
        # add the action to the simultaneous actions
        simultaneous_actions = self.simultaneous_actions + ((self.acting_players[0], action),)

        return AvalonState(self.config, self.quest_leader, self.phase, self.turn, self.round, self.done, self.good_victory, self.quest_team, self.team_votes, self.quest_votes, self.quest_results, self.roles, acting_players, simultaneous_actions)

    def get_assassin(self):
        '''
        Returns the assassin
        '''
        for i, role in enumerate(self.roles):
            if role == 7:
                return i
        
        raise ValueError('No assassin found')

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
    
    def __str__(self):
        return self.gen_state_description(with_hidden_info=True)

    def gen_state_description(self, with_hidden_info=True):
        state = self
        if with_hidden_info:
            hidden_info = f'''- The roles of the players in order are {state.get_roles_in_str_list()} with sides {state.get_is_good()} (True for Good, False for Evil)
            '''
        else:
            hidden_info = ''

        if state.phase == 0:
            phase_description = f'''- The current phase of the game is the team selection phase
            - The current leader is player {state.quest_leader}
            '''
        elif state.phase == 1:
            phase_description = f'''- The current phase of the game is the team voting phase
            - The current leader is player {state.quest_leader} who selected {state.quest_team} as the quest team
            '''
        elif state.phase == 2:
            phase_description = f'''- The current phase of the game is the quest voting phase
            - The team {state.quest_team} was approved with {state.quest_votes.count(True)} votes for and {state.quest_votes.count(False)} votes against
            '''
        elif state.phase == 3:
            phase_description = f'''- The current phase of the game is the assassination phase
            - The assassin is player {state.get_assassin()}
            '''
        else:
            raise ValueError('Invalid phase')
        
        state_description = f'''The current state of the game is as follows:
        - The number of players in the game is: {state.config.num_players}
        - This is the quest number {state.turn} which requires {state.config.num_players_for_quest[state.turn]} players and {state.config.num_fails_for_quest[state.turn]} fails to fail 
        - This is the {state.round} round of discussion
        - The previous results for the quest were {state.quest_results} (True for Success, False for Fail)
        ''' + hidden_info + phase_description
        return state_description

class AvalonInformationSet(StateTemplate):
    
    def __init__(self, config: AvalonBasicConfig, quest_leader: int, phase: int, turn: int, round: int, done: bool, quest_team: tuple[int, ...], team_votes: tuple[bool, ...], quest_results: tuple[bool, ...], known_sides: tuple[int, ...], self_role: int, self_player: int):
        '''
        '''
        self.config = config
        self.quest_leader = quest_leader
        self.phase = phase
        self.turn = turn
        self.round = round
        self.done = done
        self.quest_team = quest_team
        self.team_votes = team_votes
        self.quest_results = quest_results
        self.known_sides = known_sides
        self.self_role = self_role
        self.self_player = self_player

        # make sure all elements in here are basic types or tuple, frozenset to make it hashable and easily comparable
        id = tuple([self.quest_leader, self.phase, self.turn, self.round, self.done,
                    self.quest_team, self.team_votes,
                    self.quest_results, self.known_sides, self.self_role, self.self_player])
        super().__init__(id)

    def get_tuple(self):
        return (self.quest_leader, self.phase, self.turn, self.round, self.done, self.quest_team, self.team_votes, self.quest_results, self.known_sides, self.self_role, self.self_player)
 
    def __str__(self):
        return self.gen_str_description()

    def gen_str_description(self) -> str:
        '''
        Generate string description of the information set
        '''
        if self.phase == 0:
            phase_description = f'''This is the team selection phase. The current leader is player {self.quest_leader}
            '''
            if self.round != 0 or self.turn != 0:
                phase_description += f'''The previous team {self.quest_team} was approved by players {[index for index, value in enumerate(self.team_votes) if value == True]} and rejected by players {[index for index, value in enumerate(self.team_votes) if value == False]}
                '''
        elif self.phase == 1:
            phase_description = f'''This is the team voting phase. The current leader is player {self.quest_leader} who selected {self.quest_team} as the quest team
            '''
        elif self.phase == 2:
            phase_description = f'''This is the quest voting phase. The team {self.quest_team} was approved with {self.team_votes.count(True)} votes for and {self.team_votes.count(False)} votes against
            '''
        elif self.phase == 3:
            phase_description = f'''This is the assassination phase.
            '''
        else:
            raise ValueError('Invalid phase')
        # print("known_sides", self.known_sides)
        # print("np.where(self.known_sides == 1 or True)", np.where(self.known_sides == 1 or True))
        information_set_description = f'''The current state of the game is as follows:
        - There are {self.config.num_players} players in the game, with {self.config.num_good} good players and {self.config.num_evil} evil players
        - We are on quest number {self.turn}, which requires {self.config.num_players_for_quest[self.turn]} players and {self.config.num_fails_for_quest[self.turn]} fails to fail
        - This is round {self.round} of discussion, 5 rounds max
        - The previous results for the quest were {self.quest_results} (True for Success, False for Fail)
        - You are player {self.self_player} with role {self.config.ROLES[self.self_role]} and side {"Good" if self.config.ROLES_TO_IS_GOOD[self.self_role] else "Evil"}
        - You know that players {[index for index, value in enumerate(self.known_sides) if value == 1 or value is True]} are good and players {[index for index, value in enumerate(self.known_sides) if value == 0 or value is False]} are evil. The rest you do not know.
        ''' + phase_description

        return information_set_description
    
class AvalonInformationFunction(InformationFunction):

    actor_state_to_information_set: dict[tuple[int, AvalonState], AvalonInformationSet]

    def __init__(self, config: AvalonBasicConfig):
        self.config = config
        super().__init__()
        # we can cache previously computed results
        self.actor_state_to_information_set = dict()

    def _get_information_set(self, state: AvalonState, actor: int) -> AvalonInformationSet:
        if actor == -1:
            # actor is environment
            acting_player = state.get_acting_player()
            print("actor = ", actor)
            # assert acting_player == actor, f"Acting player {acting_player} is not the same as actor {actor}" NOTE: this is not true
            known_sides = tuple()
            self_role = -1
            information_set = AvalonInformationSet(state.config, state.quest_leader, state.phase, state.turn, state.round, state.done, state.quest_team, state.team_votes, state.quest_results, known_sides, self_role, acting_player)
        else:
            if (actor, state) not in self.actor_state_to_information_set:
                acting_player = state.get_acting_player()
                # assert acting_player == actor, f"Acting player {acting_player} is not the same as actor {actor}"
                # print("actor = ", actor)
                known_sides, self_role  = state.get_private_information(actor)
                information_set = AvalonInformationSet(state.config, state.quest_leader, state.phase, state.turn, state.round, state.done, state.quest_team, state.team_votes, state.quest_results, known_sides, self_role, actor)
                self.actor_state_to_information_set[(actor, state)] = information_set
            information_set = self.actor_state_to_information_set[(actor, state)]
        
        # self.logger.info(f"Information set for state {state} and actor {actor} is {information_set}")
        # print(f"Information set for state {state} and actor {actor} is {repr(information_set)}")
        # print("Information set", information_set)
        return information_set
    
class AvalonInformationPrior(InformationPrior):

    def __init__(self, config: AvalonBasicConfig, belief_p_is_good: NDArray, belief_p_is_merlin: NDArray, rng: np.random.Generator = np.random.default_rng()) -> None:
        super().__init__()
        self.rng = rng
        self.config = config
        self.belief_p_is_good = belief_p_is_good
        self.belief_p_is_merlin = belief_p_is_merlin
        # print(self.belief_p_is_good)
    

        # if known_role == 5: # servant
        #     # belief_p_is_good should be num_evil/ (num_players - 1)
        #     num_evil = self.config.num_evil
        #     num_players = self.config.num_players
        #     belief_p_is_good = np.ones(num_players) * num_evil / (num_players - 1)
        #     belief_p_is_good[known_player] = 1
        # else:
        #     belief_p_is_good = np.array([1 if side == 1 else 0 for side in known_sides])
        # belief_p_is_merlin = np.ones(self.config.num_players) / self.config.num_players

    def metropolis_hastings(self, n, k, p, n_iterations):
        if len(p) != n:
            raise ValueError("The length of probability vector p must match n.")
        if k > n:
            raise ValueError("k must not be greater than n.")

        # Randomly initialize the state to satisfy the constraint of exactly k 1s
        state = np.zeros(n, dtype=int)
        ones_indices = self.rng.choice(n, k, replace=False)
        state[ones_indices] = 1
        
        # samples = []
        
        for _ in range(n_iterations):
            # Propose swapping two elements, one 1 with one 0 to maintain the constraint
            ones = np.where(state == 1)[0]
            zeros = np.where(state == 0)[0]
            swap_one = self.rng.choice(ones)
            swap_zero = self.rng.choice(zeros)
            
            # Create new state with swapped values
            new_state = state.copy()
            new_state[swap_one], new_state[swap_zero] = new_state[swap_zero], new_state[swap_one]
            
            # Calculate acceptance probability
            current_prob = np.prod([p[i] if state[i] == 1 else (1 - p[i]) for i in range(n)])
            new_prob = np.prod([p[i] if new_state[i] == 1 else (1 - p[i]) for i in range(n)])
            acceptance_prob = min(1, new_prob / current_prob)
            
            # Accept or reject the new state
            if self.rng.random() < acceptance_prob:
                state = new_state
            
            # samples.append(state.copy())
        
        return state


    def sample_roles(self, known_role, known_player, known_sides, belief_p_is_good, belief_p_is_merlin,) -> tuple[int, ...]:
        # print("known_role = ", known_role)
        # print("known_player = ", known_player)
        # print("known_sides = ", known_sides)
        
        # assert that both belief_p_is_good and belief_p_is_merlin are of the correct length
        assert len(belief_p_is_good) == self.config.num_players
        assert len(belief_p_is_merlin) == self.config.num_players

        # assert that belief_p_is_merlin is a probability vector
        assert np.all(belief_p_is_merlin >= 0)
        assert np.all(belief_p_is_merlin <= 1)
        assert np.isclose(np.sum(belief_p_is_merlin), 1)

        # assert that belief_p_is_good is a probability vector
        assert np.all(belief_p_is_good >= 0)
        assert np.all(belief_p_is_good <= 1)

        # if the player is a servant we need to sample the possible sides first
        if known_role == 5: # servant
            num_evil = self.config.num_evil
            num_players = self.config.num_players 
            # remove self from belief_p_is_good
            belief_p_is_good = np.delete(belief_p_is_good, known_player)
            # sample from the possible sides
            is_good = self.metropolis_hastings(num_players-1, num_evil, belief_p_is_good, 1000)
            # insert player's own side
            is_good = np.insert(is_good, known_player, 1)
        else:
            # we assume that all sides are known to the player
            is_good = np.array(known_sides)
            # assert that there are no -1s in the known sides
            assert -1 not in is_good

        # next sample from the possible roles. recall that the player's role is known. We need to assign good roles to the good players and evil roles to the evil players

        # initialize the roles vector to all 5s
        roles = np.ones(self.config.num_players, dtype=int) * 5
        
        if known_role == 0:
            # Set the player's role to Merlin
            roles[known_player] = 0
        else:
            # Set a random good role to Merlin according to the belief
            p_is_merlin = np.where(is_good, belief_p_is_merlin, 0)
            p_is_merlin[known_player] = 0  # Player cannot be Merlin if they are not already

            # Normalize the vector
            p_is_merlin /= np.sum(p_is_merlin)

            # Sample from the vector
            merlin = self.rng.choice(self.config.num_players, p=p_is_merlin)
            roles[merlin] = 0

        # Set all is_good != 0 to Minion (6)
        roles[is_good == 0] = 6

        if known_role != 7:
            # Choose random evil player (who is not the known_player) to be the Assassin (7) uniformly
            evil_players = np.where(is_good == 0)[0]
            evil_players = evil_players[evil_players != known_player]
            assassin = self.rng.choice(evil_players)
            roles[assassin] = 7
        else:
            roles[known_player] = 7
        return tuple(roles)

    def _get_prior_state(self, information_set: AvalonInformationSet) -> AvalonState:
        '''
        The two unknowns are quest_votes and roles
        For quest_votes we can assume that nobody has voted yet
        For roles we need to sample from the possible roles
        '''
        # self.logger.debug(f"Getting prior state for information set {information_set}")
        known_player = information_set.self_player
        known_role = information_set.self_role
        known_sides = information_set.known_sides
        
        sampled_roles = self.sample_roles(known_role, known_player, known_sides, self.belief_p_is_good, self.belief_p_is_merlin)
        
        state = AvalonState.init_from_state_tuple(config=self.config, quest_leader=information_set.quest_leader, phase=information_set.phase, turn=information_set.turn, round=information_set.round, done=information_set.done, good_victory=False, quest_team=information_set.quest_team, team_votes=information_set.team_votes, quest_votes=tuple(), quest_results=tuple(), roles=sampled_roles)
        self.logger.info(f"Prior state for information set {information_set} is {state}")
        # print(f"Prior state for information set {repr(information_set)} is {state}")
        return state

class AvalonActorActionEnumerator(ActorActionEnumerator):

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

        # get all possible permutations of roles as tuples
        permutations_of_roles = list(itertools.permutations([role_id for role_id, _, _ in avalon_env.get_roles()]))
        self.permutations_of_roles = set([tuple(permutation) for permutation in permutations_of_roles])

    def _enumerate(self, state: AvalonState) ->tuple[Optional[int], set]:
        '''
        Enumerates the actors for the given state

        Args:
            state: current state

        Returns:
            actor: actor to enumerate actions for
            actions: set of actions
        '''

        if state.done:
            actor = None
        else:
            actor = state.get_acting_player()
        print(f"Actor is {actor}")
        # if state.roles are all -1, then we need to assign roles
        if state.phase == -1:
            actions = self.permutations_of_roles

        elif state.phase == 0:
            turn = state.turn
            actions = self.player_combinations[self.num_players_per_quest[turn]]

        elif state.phase == 1:
            actions = {0, 1}

        elif state.phase == 2:
            actions = {0, 1}

        elif state.phase == 3:
            actions = self.all_players

        else:
            raise ValueError('Invalid phase')

        return actor, actions
    
    @staticmethod
    def parse_str_to_action(action_str: str) -> Hashable:
        # recall that actions are either frozenset of players or int
        # if single character, then it is an int
        if len(action_str) == 1:
            return int(action_str)
        # otherwise it is a frozenset of players with string form '{1, 2, 3}'
        else:
            fs = AvalonActorActionEnumerator.string_to_frozenset(action_str)
            return fs

    @staticmethod
    def string_to_frozenset(s):
        # Trim the curly braces
        trimmed = s.strip('{}')
        
        # Check if the trimmed string is empty, return an empty frozenset
        if not trimmed:
            return frozenset()
        
        # Split the string by comma and convert each item to an integer
        elements = trimmed.split(',')
        element_set = frozenset(int(item.strip()) for item in elements)
        
        return element_set

        

class AvalonSpeakerEnumerator(SpeakerEnumerator):
    def __init__(self, avalon_env: AvalonGameEnvironment):
        super().__init__()
        self.config = avalon_env.config
        

    def _enumerate(self, state: AvalonState) -> tuple:
        '''
        Players get to discuss in the team selection phase, before any player has committed to any actions

        The speaker order should start with the quest leader, go to the right, and loop back to the quest leader
        '''
        if state.phase == 0:
            quest_leader = state.quest_leader
            # return tuple of (quest_leader, quest_leader +1 , ..., quest_leader)
            return tuple([quest_leader] + [(quest_leader + i) % self.config.num_players for i in range(1, self.config.num_players)])
        else:
            return tuple()

class AvalonTransitor(ForwardTransitor):

    def __init__(self, env: AvalonGameEnvironment):
        self.env = env
        super().__init__()


    def _transition(self, state: AvalonState, action: Hashable) -> Tuple[AvalonState, dict[int, float]]:
        '''
        Transits to the next state given the current state and action

        Args:
            state: current state
            actions: actions taken by the actors

        Returns:
            next_state: next state, reward, notes
        '''

        # print('state = ', state.id)
        
        acting_player = state.get_acting_player()
        acting_action = action
        reward = dict()
        for i in range(self.env.config.num_players):
            reward[i] = 0
        
        # if we need to assign roles, do it now
        if state.phase == -1:
            new_roles = action
            # print('new_roles = ', new_roles)
            next_state = AvalonState.init_from_state_tuple(state.config, state.quest_leader, 0, 0, 0, False, False, tuple(), tuple(), tuple(), tuple(), new_roles)
            return next_state, reward
        # first make sure we have all the simultaneous actions
        elif state.phase == 1:
            # if actions is not the same length as the number of players, then we need to wait for more actions
            # return copy of state but where acting player is the next player
            if len(state.simultaneous_actions) + 1 < self.env.config.num_players:
                next_state = state.next_simulaneous_state_copy(acting_action)
                return next_state, reward
        elif state.phase == 2:
            # if actions is not the same length as the quest team, then we need to wait for more actions
            # return copy of state but where acting player is the next player on the quest team
            if len(state.simultaneous_actions) + 1 < len(state.quest_team):
                next_state = state.next_simulaneous_state_copy(acting_action)
                return next_state, reward
            
        # combine state.simultaneous_actions with actions to get the full action set
        all_actions = {**dict(state.simultaneous_actions), acting_player: acting_action}
        
        # otherwise use env to transition

        # first extract all relevant information from the state to the environment engine
        self.env.config = state.config
        self.env.quest_leader = state.quest_leader
        self.env.phase = state.phase
        self.env.turn = state.turn
        self.env.round = state.round
        self.env.done = state.done
        self.env.good_victory = state.good_victory
        self.env.quest_team = frozenset(state.quest_team)
        self.env.team_votes = list(state.team_votes)
        self.env.quest_votes = list(state.quest_votes)
        self.env.quest_results = list(state.quest_results)
        self.env.roles = np.array(state.roles)
        self.env.is_good = np.array(state.get_is_good())
        # state.assassin


        if self.env.phase == 0:
            action = all_actions[state.quest_leader]
            self.env.choose_quest_team(action, self.env.quest_leader)
            # self.env.phase, self.env.done, self.env.quest_leader = 

        elif self.env.phase == 1:
            action = list(all_actions.values())
            self.env.gather_team_votes(action)

            # self.env.phase, self.env.done, team_votes_result = 

        elif self.env.phase == 2:
            action = list(all_actions.values())
            #print(AvalonState(self.env))
            self.env.gather_quest_votes(action)
            # self.env.phase, self.env.done, quest_vote_result, num_fails = 


        elif self.env.phase == 3:
            player = state.get_assassin()
            action = all_actions[player]
            self.env.choose_assassination_target(player, action)

            # self.env.phase, self.env.done, self.env.good_victory = 
        
        # now extract all relevant information from the environment engine to the next state
        next_state = AvalonState.init_from_env(self.env)

        # print('next_state = ', next_state.id)
        # print('done = ', self.env.done)

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

        return (next_state, reward)

class AvalonLLMFunctionalValueHeuristic(ValueHeuristic):
    '''
    Functional value heuristic for LLMs
    '''

    EVAL_TEST_STATES = [
        {'quest_leader': 0, 'phase': 3, 'turn': 3, 'round': 0, 'quest_team': (3, 4), 'historical_team_votes': (0, 0, 0, 1, 1), 'historical_quest_results': (True, True, True), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Minion', 'Merlin', 'Assassin', 'Servant', 'Servant'], 'is_good': [False, True, False, True, True]}, 
        {'quest_leader': 0, 'phase': 2, 'turn': 2, 'round': 0, 'quest_team': (3, 4), 'historical_team_votes': (1, 0, 0, 1, 1), 'historical_quest_results': (True, True), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Minion', 'Merlin', 'Assassin', 'Servant', 'Servant'], 'is_good': [False, True, False, True, True]}, 
        {'quest_leader': 4, 'phase': 0, 'turn': 1, 'round': 0, 'quest_team': (1, 4), 'historical_team_votes': (0, 1, 0, 0, 1), 'historical_quest_results': (True,), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Minion', 'Merlin', 'Assassin', 'Servant', 'Servant'], 'is_good': [False, True, False, True, True]},
        {'quest_leader': 0, 'phase': 0, 'turn': 0, 'round': 1, 'quest_team': (1, 2), 'historical_team_votes': (1, 0, 1, 0, 0), 'historical_quest_results': tuple(), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Minion', 'Merlin', 'Assassin', 'Servant', 'Servant'], 'is_good': [False, True, False, True, True]},
        {'quest_leader': 2, 'phase': 0, 'turn': 0, 'round': 1, 'quest_team': (1, 4), 'historical_team_votes': (1, 0, 1, 0, 0), 'historical_quest_results': (), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Minion', 'Merlin', 'Servant', 'Assassin', 'Servant'], 'is_good': [False, True, True, False, True]},
        {'quest_leader': 1, 'phase': 1, 'turn': 0, 'round': 0, 'quest_team': (1, 2), 'historical_team_votes': (), 'historical_quest_results': (), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Minion', 'Assassin', 'Servant', 'Merlin', 'Servant'], 'is_good': [False, False, True, True, True]},
        {'quest_leader': 2, 'phase': 0, 'turn': 0, 'round': 1, 'quest_team': (1, 4), 'historical_team_votes': (1, 0, 1, 0, 0), 'historical_quest_results': (), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Assassin', 'Merlin', 'Minion', 'Servant', 'Servant'], 'is_good': [False, True, False, True, True]},
        {'quest_leader': 2, 'phase': 0, 'turn': 0, 'round': 1, 'quest_team': (1, 4), 'historical_team_votes': (1, 0, 1, 0, 0), 'historical_quest_results': (), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Assassin', 'Servant', 'Servant', 'Merlin', 'Minion'], 'is_good': [False, True, True, True, False]},
        {'quest_leader': 0, 'phase': 1, 'turn': 0, 'round': 0, 'quest_team': (1, 4), 'historical_team_votes': (), 'historical_quest_results': (), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Minion', 'Servant', 'Merlin', 'Assassin', 'Servant'], 'is_good': [False, True, True, False, True]},
        {'quest_leader': 3, 'phase': 1, 'turn': 0, 'round': 1, 'quest_team': (1, 4), 'historical_team_votes': (0, 0, 0, 0, 0), 'historical_quest_results': (), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Merlin', 'Assassin', 'Servant', 'Servant', 'Minion'], 'is_good': [True, False, True, True, False]},
        {'quest_leader': 1, 'phase': 2, 'turn': 0, 'round': 0, 'quest_team': (3, 4), 'historical_team_votes': (0, 0, 0, 0, 1), 'historical_quest_results': (), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Assassin', 'Minion', 'Merlin', 'Servant', 'Servant'], 'is_good': [False, False, True, True, True]},
        {'quest_leader': 2, 'phase': 1, 'turn': 1, 'round': 0, 'quest_team': (0, 1, 4), 'historical_team_votes': (1, 1, 1, 0, 1), 'historical_quest_results': (True,), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Servant', 'Servant', 'Minion', 'Assassin', 'Merlin'], 'is_good': [True, True, False, False, True]},
        {'quest_leader': 2, 'phase': 0, 'turn': 0, 'round': 2, 'quest_team': (3, 4), 'historical_team_votes': (0, 0, 0, 0, 1), 'historical_quest_results': (), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Assassin', 'Servant', 'Minion', 'Merlin', 'Servant'], 'is_good': [False, True, False, True, True]},
        {'quest_leader': 0, 'phase': 2, 'turn': 1, 'round': 0, 'quest_team': (0, 3, 4), 'historical_team_votes': (0, 0, 0, 0, 1), 'historical_quest_results': (False,), 'players': {0, 1, 2, 3, 4}, 'num_good': 3, 'num_participants_per_quest': [2, 3, 2, 3, 3], 'num_fails_per_quest': [1, 1, 1, 1, 1], 'roles': ['Servant', 'Servant', 'Assassin', 'Minion', 'Merlin'], 'is_good': [True, True, False, False, True]},  

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

    # @staticmethod
    # def generate_seed_function(model: LLMModel, prompt_gen: PromptGenerator) -> str:
    #     '''
    #     Generates a seed function from the model
    #     '''
    #     prompt = prompt_gen.gen_seed_thought_prompt()
    #     thought = model.generate(prompt)
    #     prompt = prompt_gen.gen_seed_function_prompt(thought)
    #     function_str = model.generate(prompt)
    #     return function_str[0]
    
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



        # input_dict['roles'] = {player:role for player, role in enumerate(state.get_roles_in_str_list())}
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
                    # print('testing', state)
                    player_to_score, notes = llm_evaluate(state)
                    # assert that player_to_score is a dictionary
                    assert isinstance(player_to_score, dict)
                    # assert that all values in player_to_score are numbers
                    assert all(isinstance(value, (int, float)) for value in player_to_score.values())
                    # assert that notes is a dictionary
                    assert isinstance(notes, dict)

                    assert set(player_to_score.keys()) == state['players']

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
                
                player_to_score, notes = llm_evaluate(state)
                # assert that player_to_score is a dictionary
                assert isinstance(player_to_score, dict)
                # assert that all values in player_to_score are numbers
                assert all(isinstance(value, (int, float)) for value in player_to_score.values())
                # assert that notes is a dictionary
                assert isinstance(notes, dict)

                assert set(player_to_score.keys()) == state['players']

            return True

    def _evaluate(self, state: AvalonState) -> tuple[dict, dict]:
        # Prepare input
        input_dict = AvalonLLMFunctionalValueHeuristic.convert_state_to_input(state)

        # use the function to calculate the value
        try:
            winrates, notes = self._llm_evaluate(input_dict)
            # logging.info(f"state dictionary: {str(input_dict)}")
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