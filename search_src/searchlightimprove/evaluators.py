from .headers import Evaluator
from search_src.searchlight.gameplay.simulators import GameSimulator
from search_src.searchlight.headers import *
from search_src.searchlight.gameplay.agents import SearchAgent
from search_src.searchlight.algorithms.mcts_search import SMMonteCarlo
from search_src.searchlight.datastructures.graphs import ValueGraph2
from search_src.searchlight.datastructures.adjusters import PUCTAdjuster
from search_src.searchlight.datastructures.estimators import UtilityEstimatorLast
from search_src.searchlight.classic_models import RandomRolloutValueHeuristic

import numpy as np
from collections import defaultdict
import itertools

class DummyEvaluator(Evaluator):
    '''
    Dummy evaluator for testing
    '''
    def __init__(self):
        super().__init__()

    def evaluate(self, objects: list[Any]) -> tuple[list[float],list]:
        '''
        Evaluates a collection of functions

        Args:
            functions: collection of functions to evaluate

        Returns:
            scores: scores of the functions
            notes: notes for each function
        '''
        return [0.5]*len(objects), [dict()]*len(objects)

class SimulateGameEvaluator(Evaluator):
    '''
    Abstract class.

    Simulates a game using the given agents returns the scores based on the outcome of the game
    '''
    def __init__(self, simulator: GameSimulator, ):
        super().__init__()
        self.simulator = simulator

    def evaluate(self, objects: list[Any]) -> tuple[list[float], list]:
        raise NotImplementedError

class SimulateSearchGameEvaluator(SimulateGameEvaluator):
    '''
    Abstract class.

    Simulates a game using the given agents returns the scores based on the outcome of the game

    Assumes that the agents are identical
    '''
    def __init__(self, simulator: GameSimulator,
                 num_batch_runs: int = 10,
                 players = {0, 1}, 
                 rng = np.random.default_rng(), 
                 stochastic_combinations = True):
        super().__init__(simulator=simulator,)
        self.players = players
        self.num_batch_runs = num_batch_runs
        self.random_agent = RandomAgent(rng)
        self.random_agent.set_player(-1)
        self.rng = rng
        self.stochastic_combinations = stochastic_combinations

    def set_num_batch_runs(self, num_batch_runs: int):
        self.num_batch_runs = num_batch_runs

    def get_num_batch_runs(self):
        return self.num_batch_runs

    def evaluate(self, objects: list[Any]) -> tuple[list[float], list]:
        # create a agents
        agents = self.create_agents(objects)

        # evaluate the agents
        scores, notes = self.evaluate_agents(agents)

        return scores, notes

    def evaluate_agents(self, agents: list[Agent]) -> tuple[list[float], list]:

        # check that len(self.players) <= len(agents)
        assert len(self.players) <= len(agents)

        # check that all the agents are different objects
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    assert agent1 != agent2

        if not self.stochastic_combinations:
            # enumerate all possible permutations of len(players) of range(len(functions))
            all_permutations = list(itertools.permutations(range(len(agents)), len(self.players)))

        # enumerate all possible permutations of len(players) of range(len(functions))
        # all_combinations = list(itertools.combinations(range(len(agents)), len(self.players)))

        # simulate games for each combination
        score_per_agent = [0.0 for _ in range(len(agents))]
        notes_per_agent = [defaultdict(list) for _ in range(len(agents))]
        num_plays_per_agent = [0 for _ in range(len(agents))]
        
        # loop through number of batch runs
        for _ in range(self.num_batch_runs):
            
            def play_out_combination(combination):
                # assign agents to players
                agent_subset = {actor: agents[combination[i]] for i, actor in enumerate(self.players)}
                # set_player the agents to correct actor
                for actor, agent in agent_subset.items():
                    agent.set_player(actor)

                # add random agent for environment
                agent_subset[-1] = self.random_agent

                # print out agent.player for debugging
                # print('agent player', {actor: agent.player for actor, agent in agent_subset.items()})

                # check that all the agents are different objects
                for i, agent1 in enumerate(agent_subset.values()):
                    for j, agent2 in enumerate(agent_subset.values()):
                        if i != j:
                            assert agent1 != agent2

                # log agent subset for debugging
                # self.logger.info(f'agent_subset: {agent_subset}')


                score, trajectory = self.simulator.simulate_game(agent_subset)

                # self.logger.info(f'score: {score}')
                score = dict(score) # in case it is a defaultdict
                # TODO: we also want to include execution error feedback here

                # everything below is for recording feedback

                # get the heuristic for each state along the trajectory
                # and the search estimate
                heuristics_trajectory = defaultdict(list)
                heuristics_score = defaultdict(list)
                estimate_trajectory = defaultdict(list)
                score_trajectory = defaultdict(list)
                for _, _, state in trajectory:
                    for i, actor in enumerate(self.players):
                        if actor != -1:
                            agent = agent_subset[actor]
                            if type(agent) == SearchAgent:
                                agent_id = combination[i]
                                graph = agent.get_graph()
                                node = graph.get_node(state)
        
                                # get the heuristic feedback
                                if node is None:
                                    self.logger.debug('State {} not found in graph'.format(state))
                                    # usually this is normal and is because the agent did not explore the state
                                    heuristics_trajectory[agent].append(dict())
                                    estimate_trajectory[agent].append(search_estimate)
                                    heuristics_score[agent].append(dict())
                                    # TODO: this is a hack, we should get rid of problematic states
                                else:
                                    # print('node notes', node.notes)
                                    # get the heuristic feedback
                                    if 'heuristic_feedback' in node.notes:
                                        heuristics_trajectory[agent].append(node.notes['heuristic_feedback'])
                                    else:
                                        heuristics_trajectory[agent].append(dict())

                                    # get the search estimate
                                    search_estimate = {player: graph.get_estimated_value(node, player) for player in self.players}
                                    estimate_trajectory[agent].append(search_estimate)

                                    # get the heuristic score
                                    if 'heuristic_score' in node.notes:
                                        heuristics_score[agent].append(dict(node.notes['heuristic_score']))
                                    else:
                                        heuristics_score[agent].append(dict())

                                # append endgame score
                                score_trajectory[agent].append(score)


                
                # print('heuristics_trajectory', heuristics_trajectory)
                
                # add relevant stats for each agent in the combination
                for i, actor in enumerate(self.players):
                    agent_id = combination[i]
                    agent = agent_subset[actor]
                    score_per_agent[agent_id] += score[actor]
                    num_plays_per_agent[agent_id] += 1
                    # combine all the trajectory data into a single dictionary for that agent
                    trajectory_data = {'trajectory': trajectory, 'heuristics_score_trajectory': heuristics_score[agent], 'heuristics_trajectory': heuristics_trajectory[agent], 'search_trajectory': estimate_trajectory[agent], 'score_trajectory': score_trajectory[agent]}
                    notes_per_agent[agent_id]['trajectory_data'].append(trajectory_data)
            
            if not self.stochastic_combinations:
                # enumerate all possible permutations of len(players) of range(len(functions))
                
                for permutation in all_permutations:
                    # print the combination for debugging
                    # print('combination', combination)

                    # print self.players for debugging
                    # print('self.players', self.players)
                    play_out_combination(permutation)
            else:
                # sample a random permutation of len(players) of range(len(functions))
                permutation = self.rng.permutation(len(agents))[:len(self.players)]
                play_out_combination(permutation)
        # average the scores
        for i in range(len(score_per_agent)):
            if num_plays_per_agent[i] == 0:
                score_per_agent[i] = 0.0
                self.logger.warning(f'Agent {i} did not play any games')
            else:
                score_per_agent[i] /= num_plays_per_agent[i]
        # if self.stochastic_combinations:
        #     score_per_agent = [score / self.num_batch_runs for score in score_per_agent]
        # else:
        #     score_per_agent = [score / (self.num_batch_runs * len(all_permutations)) for score in score_per_agent]

        # log len(notes_per_agent[agent_id]['trajectory_data']) for agent 0 
        # self.logger.info(f'len(notes_per_agent[0][trajectory_data]): {len(notes_per_agent[0]["trajectory_data"])}')
        # now log the trajectory data for agent 0
        # self.logger.info(f'notes_per_agent[0]["trajectory_data"]: {notes_per_agent[0]["trajectory_data"]}')

        # figure out what notes we want to return
        # should include the following:
        # - trajectories of all games played
        # - execution error feedback
        # - what the heuristic function evaluated to for each state (node) and related components
        
        return score_per_agent, notes_per_agent

    
    def create_agents(self, objects: list[Any]) -> list[Agent]:
        '''
        Creates agents for each function
        Note that we assume the functions are executable
        This usually only includes player agents, not the environment agent 

        Args:
            functions: list of functions

        Returns:
            agents: list of agents
        '''
        raise NotImplementedError
    
