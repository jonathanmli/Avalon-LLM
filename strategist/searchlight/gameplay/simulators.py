from collections import defaultdict
from typing import Any, Tuple, Set, Optional
from ..utils import AbstractLogged

from strategist.searchlight.headers import *
from tqdm import tqdm
import logging
import random

class GameSimulator(AbstractLogged):
    '''
    Simulates the agents playing a game and returns the scores based on the outcome of the game

    The simulator uses the transitor to transition between states, the actor enumerator to enumerate the actors for a given state, and the action enumerator to enumerate the legal actions
    '''

    def __init__(self, transitor: ForwardTransitor,
                    actor_action_enumerator: ActorActionEnumerator,
                    information_function: Optional[InformationFunction] = None,
                    start_state: Hashable = None,
                    rng: np.random.Generator = np.random.default_rng()):
            super().__init__()
            self.transitor = transitor
            self.actor_action_enumerator = actor_action_enumerator
            self.start_state = start_state
            self.information_function = information_function
            self.rng = rng

    
    def simulate_game(self, agents: dict[Any, Agent], start_state: Hashable = None, display: bool = False) -> tuple[dict[Any, float], list[tuple[dict, dict, Hashable]]]:
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
            
            # log the state for debugging
            # self.logger.info(f'State: {state}')
            
            # get the current actors
            actor, legal_actions = self.actor_action_enumerator.enumerate(state)
            # if there are no actors, then the game is done
            if actor is None:
                done = True
                break


            
            if self.information_function is not None:
                information_set = self.information_function.get_information_set(state, actor)
                action = agents[actor].act(information_set, legal_actions)
            else:
                # self.logger.debug(f'Agent: {agents[actor]}')
                action = agents[actor].act(state, legal_actions)
                # actions[actor] = random.choice(list(allowed_actions))
                # FIXME: random action works here, but agents[actor].act changes agents somehow?
                # I suspect that the transitor/action_enumerator/or actor_enumerator is not working correctly
                # and the problem is that there are some connections when we use the same instance of the transitor/action_enumerator/actor_enumerator for each agent and the simulator

            # print agents.keys()
            # self.logger.debug(f'Agents keys after actions: {agents.keys()}')
            # transition to the next state
            state, rewards = self.transitor.transition(state, action, actor)
            # done = state.is_done()
            # append to trajectory
            trajectory.append((action, rewards, state))
            # add rewards to sum_scores
            for actor, reward in rewards.items():
                sum_scores[actor] += reward

            if display:
                self.logger.info(f'State: {state}')
                self.logger.info(f'Actor: {actor}')
                self.logger.info(f'Action: {action}')
                self.logger.info(f'Rewards: {rewards}')
        return sum_scores, trajectory
    
    def simulate_games(self, agents: dict[Any, Agent], num_games: int, start_state: Hashable = None, display: bool = False,) -> tuple[dict[Any, float], list]:
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
                # if random_initialize_start:
                #     start_state.initial_state_randomize(self.rng)
                scores, trajectory = self.simulate_game(agents, start_state)
                trajectories.append(trajectory)
                for actor, score in scores.items():
                    sum_scores[actor] += score
        else:
            for _ in tqdm(range(num_games)):
                # print agents.keys()
                # self.logger.debug(f'Agents keys 2: {agents.keys()}')
                # if random_initialize_start:
                #     start_state.initial_state_randomize(self.rng)
                scores, trajectory = self.simulate_game(agents, start_state, display=display)
                trajectories.append(trajectory)
                for actor, score in scores.items():
                    sum_scores[actor] += score

        average_scores = {actor: score/num_games for actor, score in sum_scores.items()}

        return average_scores, trajectories
    
class DialogueGameSimulator(GameSimulator):
    '''
    Simulates the agents playing a dialogue game and returns the scores based on the outcome of the game

    The simulator uses the transitor to transition between states, the actor enumerator to enumerate the actors for a given state, and the action enumerator to enumerate the legal actions
    '''
    def __init__(self, transitor: ForwardTransitor, actor_action_enumerator: ActorActionEnumerator, speaker_enumerator: SpeakerEnumerator, information_function: Optional[InformationFunction] = None, start_state: Hashable = None, rng: np.random.Generator = np.random.default_rng()):
        super().__init__(transitor=transitor, actor_action_enumerator=actor_action_enumerator, information_function=information_function, start_state=start_state, rng=rng)
        self.speaker_enumerator = speaker_enumerator

    def slice_out_new_dialogue(self, dialogue_history: list[tuple[int, str]], player: int) -> list[tuple[int, str]]:
        '''
        Returns a list of dialogue that is new to the player (i.e. from the last time the player spoke, or the beginning of the dialogue if the player has not spoken yet)
        '''
        new_dialogue = dialogue_history
        for i, (speaker, utterance) in enumerate(dialogue_history):
            if speaker == player:
                new_dialogue = dialogue_history[i+1:]
                break
        return new_dialogue

    def simulate_game(self, agents: dict[Any, DialogueAgent], start_state: Hashable = None, display: bool = False) -> tuple[dict[Any, float], list[tuple[dict, dict, Hashable]]]:
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
            
            # log the state for debugging
            # self.logger.info(f'State: {state}')
            
            # get the current actors
            actor, legal_actions = self.actor_action_enumerator.enumerate(state)
            # print("Actor: ", actor)
            # print("State: ", state)
            # if there are no actors, then the game is done
            if actor is None:
                done = True
                break

            # see if there is a discussion round
            speakers = self.speaker_enumerator.enumerate(state)
            # print('speakers', speakers)
            dialogue_history: list[tuple[int, str]] = []
            # see if speakers is empty
            if speakers:
                for speaker in speakers:
                    # first have the speaker observe_dialogue and new dialogue (dialogue produced from the last point they spoke (exclusive))
                    new_dialogue = self.slice_out_new_dialogue(dialogue_history, speaker)
                    if self.information_function is not None:
                        information_set = self.information_function.get_information_set(state, speaker)
                        print(f"Speaker {speaker} is thinking.")
                        agents[speaker].observe_dialogue(state=information_set, 
                    new_dialogue=new_dialogue)
                        # then have the speaker produce an utterance
                        utterance = agents[speaker].produce_utterance(state=information_set)
                        dialogue_history.append((speaker, utterance))
                        # print(f"Speaker {speaker} done speaking.")
                    else:
                        agents[speaker].observe_dialogue(state=state, 
                    new_dialogue=new_dialogue)
                        # then have the speaker produce an utterance
                        utterance = agents[speaker].produce_utterance(state=state)
                        dialogue_history.append((speaker, utterance))
                
                print("Discussion finished.")
                # update all agents with the dialogue
                for _actor, agent in agents.items():
                    new_dialogue = self.slice_out_new_dialogue(dialogue_history, _actor)
                    if self.information_function is not None:
                        information_set = self.information_function.get_information_set(state, _actor)
                        agents[_actor].observe_dialogue(state=information_set,new_dialogue=new_dialogue)
                    else:
                        agent.observe_dialogue(state=state, new_dialogue=new_dialogue)

            if self.information_function is not None:
                # print("state: ", state)
                # print("actor: ", actor)
                information_set = self.information_function.get_information_set(state, actor)
                # print("Information set: ", repr(information_set))
                # print("Information set class: ", information_set.__class__)
                action = agents[actor].act(information_set, legal_actions)
            else:
                action = agents[actor].act(state, legal_actions)

            # print agents.keys()
            # self.logger.debug(f'Agents keys after actions: {agents.keys()}')
            # transition to the next state
            state, rewards = self.transitor.transition(state, action, actor)
            # done = state.is_done()
            # append to trajectory
            trajectory.append((action, rewards, state))
            # add rewards to sum_scores
            for actor, reward in rewards.items():
                sum_scores[actor] += reward

            if display:
                self.logger.info(f'State: {state}')
                self.logger.info(f'Actor: {actor}')
                self.logger.info(f'Action: {action}')
                self.logger.info(f'Rewards: {rewards}')
                self.logger.info(f'Dialogue: {dialogue_history}')
        return sum_scores, trajectory
