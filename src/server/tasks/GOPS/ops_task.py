import sys
import json
from copy import deepcopy
from typing import List, Tuple, Dict, Any

from src.server.task import Task, Session
from src.typings import TaskSampleExecutionResult, TaskOutput, SampleIndex, AgentOutputStatus, SampleStatus
from src.utils import ColorMessage

from .engine import *
from .agents.naive import NaiveGOPSAgent
from .agents.llmagent import LLMGOPSAgent
from .agents.alphabeta import AlphaBetaBot
from .agents.mcts import MCTSBot

from .wrapper import SessionWrapper
from multi_agent.typings import FakeSession

from src.typings import AgentContextLimitException

from multi_agent.proxy import MultiAgentProxy

import pyspiel
import numpy as np
from open_spiel.python.algorithms import mcts, minimax, tabular_qlearner, nash_averaging
from .utils import *

AGENT_FINDER = {
    'naive': NaiveGOPSAgent,
    'alphabeta': AlphaBetaBot,
    'llm': LLMGOPSAgent,
    'mcts': MCTSBot
}

class GOPSBench(Task):
    def __init__(self, num_games, num_turns, agent_list, **configs):
        super().__init__(**configs)

        self.num_games = num_games
        self.num_turns = num_turns
        self.agent_list = agent_list
        self.data = [i for i in range(num_games)]


    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        outputs = [None for _ in range(len(self.data))]
        for result in results:
            outputs[result.index] = result.result

        player1_wins = 0
        player2_wins = 0
        ties = 0

        player1_total_score = 0
        player2_total_score = 0

        for output in outputs:
            if output["player1_score"] > output["player2_score"]:
                player1_wins += 1
            elif output["player1_score"] < output["player2_score"]:
                player2_wins += 1
            else:
                ties += 1

            player1_total_score += output["player1_score"]
            player2_total_score += output["player2_score"]

        return {
            "player 1": self.agent_list[0],
            "player 2": self.agent_list[1],
            "winrate of player 1": player1_wins / len(self.data),
            "winrate of player 2": player2_wins / len(self.data),
            "tie rate": ties / len(self.data),
            "player 1 average score": player1_total_score / len(self.data),
            "player 2 average score": player2_total_score / len(self.data)
        }

    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.data)))

    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        assert isinstance(index, int), "Index must be an integer"
        game = pyspiel.load_game_as_turn_based("goofspiel", {"num_cards": self.num_turns})
        proxy = MultiAgentProxy(session, num_agents=2)
        sessions = [SessionWrapper(session, proxy), SessionWrapper(session, proxy)]
        proxy.initialize_sessions(sessions)
        print(proxy.session_list)
        print("PID: ", proxy.current_agent)

        # Create random state
        rng = np.random.RandomState(index)

        initial_hands = [i+1 for i in range(self.num_turns)]

        # Initialize players
        # Should let LLM be player 1 by default
        player1 = AGENT_FINDER[self.agent_list[0]](
            id      =   0,
            hand    =   deepcopy(initial_hands),
            session =   sessions[0],
            game    =   game
        )
        player2 = AGENT_FINDER[self.agent_list[1]](
            id      =   1,
            hand    =   deepcopy(initial_hands),
            session =   sessions[1],
            game    =   game
        )

        for player in [player2, player1]:
            await player.initialize()
            pid = proxy.get_next_agent()



        print(f"Welcome {player1} and {player2} to GOPS!")
        print("Player2 PID: ", proxy.current_agent)
        state = game.new_initial_state()
        round_id = 0
        move1 = -1
        move2 = -1
        score_card_left = []
        while not state.is_terminal():
            if state.is_chance_node():
                # Sample a chance event outcome.
                outcomes_with_probs = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = rng.choice(action_list, p=prob_list)
                score_card_left = list(np.array(list(action_list)) + 1)
                state.apply_action(action)
                if round_id != 0:
                    await player1.observe_round(
                        contested_points    =   contested_scores,
                        your_card           =   move1,
                        opponent_card       =   move2,
                        round_id            =   round_id
                    )
                    # pid = proxy.get_next_agent()
                    # print("Next player: ", pid)
                    await player2.observe_round(
                        contested_points    =   contested_scores,
                        your_card           =   move2,
                        opponent_card       =   move1,
                        round_id            =   round_id
                    )
                round_id += 1
            else:
                contested_scores = sum(get_score_card(state)) - sum(get_points(state))
                print("Contested Scores: ", contested_scores)
                if state.current_player() == 0:
                    # player 1's turn
                    action = await player1.step(
                        state               =    state,
                        opponent_hand       =    get_player2_hands(state),
                        contested_scores    =    contested_scores,
                        score_card_left     =    score_card_left
                    )
                    move1 = int(action) + 1
                else:
                    # player 2's turn
                    action = await player2.step(
                        state               =    state,
                        opponent_hand       =    get_player1_hands(state),
                        contested_scores    =    contested_scores,
                        score_card_left     =    score_card_left
                    )

                    move2 = int(action) + 1

                    # pid = proxy.get_next_agent()
                    # print("Next player: ", pid)

                print("Player {} takes action {} at state {}".format(state.current_player(), action, state))

                state.apply_action(action)

        # Episode is over, update return
        returns = state.returns()
        print("Player 1: {}".format(returns[0]))
        print("Player 2: {}".format(returns[1]))

        finish_reason = SampleStatus.COMPLETED

        return TaskSampleExecutionResult(status=finish_reason, result={
            "player1": self.agent_list[0],
            "player2": self.agent_list[1],
            "player1_score": int(get_points(state)[0]),
            "player2_score": int(get_points(state)[1]),
            "history of player 1": proxy.history[0],
            "history of player 2": proxy.history[1],
        })