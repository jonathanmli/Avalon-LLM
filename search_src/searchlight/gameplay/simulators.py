from collections import defaultdict
from typing import Any, Tuple, Set, Optional

from src.searchlight.headers import *
from tqdm import tqdm
import logging
import random

class GameSimulator:
    '''
    Simulates the agents playing a game and returns the scores based on the outcome of the game

    The simulator uses the transitor to transition between states, the actor enumerator to enumerate the actors for a given state, and the action enumerator to enumerate the legal actions
    '''
    _logger_number = 0  # Class variable to keep track of instances

    def __init__(self, transitor: ForwardTransitor2,
                    actor_enumerator: ActorEnumerator,
                    action_enumerator: ActionEnumerator, 
                    start_state: Optional[State] = None,
                    rng: np.random.Generator = np.random.default_rng()):
            super().__init__()
            self.transitor = transitor
            self.actor_enumerator = actor_enumerator
            self.start_state = start_state
            self.action_enumerator = action_enumerator
            self.rng = rng
            # Create a logger for each instance with the name of the class and name of the instance
            # Increment the logger number for each instance
            type(self)._logger_number += 1
            self.logger = logging.getLogger(f'{self.__class__.__name__}_{type(self)._logger_number}'
                                            
)
    
    def simulate_game(self, agents: dict[Any, Agent], start_state: Optional[State] = None) -> tuple[dict[Any, float], list[tuple[dict, dict, State]]]:
        '''
        Simulates a game using the given agents

        Args:
            agents: dictionary of from actor to agent. note that actor -1 is the environment and must be included
            start_state: state to start the game from

        Returns:
            scores: scores of all the agents except actor -1, the environment
            trajectory: trajectory of the game (state, action, reward) for each time step
        '''
        if start_state is None:
            start_state = self.start_state 

        if start_state is None:
            raise ValueError("start_state must be provided if self.start_state is None")

        trajectory = []
        sum_scores = defaultdict(float)
        state = start_state
        done = False
        while not done:
            # get the current actors
            actors = self.actor_enumerator.enumerate(state)
            # if there are no actors, then the game is done
            if not actors:
                done = True
                break
            # get the actions for each actor
            actions = dict()

            # print agents.keys()
            # self.logger.debug(f'Agents keys before actions: {agents.keys()}')

            for actor in actors:
                # print agents.keys()
                # self.logger.debug(f'Agents keys during actions: {agents.keys()}')
                allowed_actions = self.action_enumerator.enumerate(state, actor)
                # allowed_actions = {0,1,2,}
                # print agent
                # self.logger.debug(f'Agent: {agents[actor]}')
                actions[actor] = agents[actor].act(state, allowed_actions)
                # actions[actor] = random.choice(list(allowed_actions))
                # FIXME: random action works here, but agents[actor].act changes agents somehow?
                # I suspect that the transitor/action_enumerator/or actor_enumerator is not working correctly
                # and the problem is that there are some connections when we use the same instance of the transitor/action_enumerator/actor_enumerator for each agent and the simulator


            # print agents.keys()
            # self.logger.debug(f'Agents keys after actions: {agents.keys()}')
            # transition to the next state
            state, rewards, notes = self.transitor.transition(state, actions)
            # done = state.is_done()
            # append to trajectory
            trajectory.append((actions, rewards, state))
            # add rewards to sum_scores
            for actor, reward in rewards.items():
                sum_scores[actor] += reward
        return sum_scores, trajectory
    
    def simulate_games(self, agents: dict[Any, Agent], num_games: int, start_state: Optional[State] = None, display: bool = False, random_initialize_start=False) -> tuple[dict[Any, float], list]:
        '''
        Simulates a game using the given agents for num_games

        Args:
            agents: dictionary of from actor to agent
            num_games: number of games to simulate
            start_state: state to start the game from
            display: whether to display the progress bar

        Returns:
            average_scores: average scores of all the agents except actor -1, the environment
            trajectories: list of trajectories of the game (state, action, reward) for each time step
        '''

        if start_state is None:
            start_state = self.start_state 
        if start_state is None:
            raise ValueError("start_state must be provided if self.start_state is None")
        
        trajectories = []
        sum_scores = defaultdict(float)
        
        if not display:
            for _ in range(num_games):
                if random_initialize_start:
                    start_state.initial_state_randomize(self.rng)
                scores, trajectory = self.simulate_game(agents, start_state)
                trajectories.append(trajectory)
                for actor, score in scores.items():
                    sum_scores[actor] += score
        else:
            for _ in tqdm(range(num_games)):
                # print agents.keys()
                # self.logger.debug(f'Agents keys 2: {agents.keys()}')
                if random_initialize_start:
                    start_state.initial_state_randomize(self.rng)
                scores, trajectory = self.simulate_game(agents, start_state)
                trajectories.append(trajectory)
                for actor, score in scores.items():
                    sum_scores[actor] += score

        average_scores = {actor: score/num_games for actor, score in sum_scores.items()}

        return average_scores, trajectories