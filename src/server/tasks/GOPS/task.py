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
from .agents.smminimax import SMMinimaxCustomBot

from .wrapper import GOPSSessionWrapper
from multi_agent.typings import FakeSession

from src.typings import AgentContextLimitException

from multi_agent.proxy import MultiAgentProxy

AGENT_FINDER = {
    'naive': NaiveGOPSAgent,
    'llm': LLMGOPSAgent,
    'smminimax': SMMinimaxCustomBot
}

class GOPSBench(Task):
    def __init__(self, num_games, num_turns, agent_list, **configs):
        super().__init__(**configs)

        self.num_games = num_games
        self.num_turns = num_turns
        self.agent_list = agent_list

        self.data = [0 for _ in range(num_games)]


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
        print("test-1")
        proxy = MultiAgentProxy(session, num_agents=2)
        sessions = [GOPSSessionWrapper(session, proxy), GOPSSessionWrapper(session, proxy)]
        proxy.initialize_sessions(sessions)
        print(proxy.session_list)

        config = GOPSConfig(self.num_turns)
        env = GOPSEnvironment(config)

        (done, score_card, contested_points) = env.reset()

        # Initialize players
        # Should let LLM be player 1 by default
        player1 = AGENT_FINDER[self.agent_list[0]](
            id      =   0,
            hand    =   deepcopy(env.player1_hand),
            session =   sessions[0]
        )
        print("test0")
        player2 = AGENT_FINDER[self.agent_list[1]](
            id      =   1,
            hand    =   deepcopy(env.player2_hand),
            session =   sessions[1]
        )
        print("0.0")

        for player in [player1, player2]:
            await player.initialize()
            pid = proxy.get_next_agent()

        print(f"Welcome {player1} and {player2} to GOPS!")
        state = ''
        prize_cards = []
        player_cards = []
        opponent_cards = []
        while not done:
            print(f"Current score: {player1}: {env.player1_score}, {player2}: {env.player2_score}")
            print(f"Current contested points: {contested_points}, current contested score card: {score_card}")

            print(f"{player1}, play a card out of {env.player1_hand}")

            score_card_left = list(env.get_score_card_deck())
            round_id = env.get_current_turn()

            reserved_p1_hand = list(player1.hand)
            reserved_p2_hand = list(player2.hand)

            prize_cards.append(score_card)

            print("test1")

            move1 = await player1.step(
                prize_cards = prize_cards,
                player_cards = player_cards,
                opponent_cards = opponent_cards,
            )
            print("move1: ", move1)

            move2 = await player2.step(
                prize_cards = prize_cards,
                player_cards = opponent_cards,
                opponent_cards = player_cards,
            )
            print("move2: ", move2)

            print("test2")

            # move1 = await player1.step(
            #     state               =    state,
            #     opponent_hand       =    reserved_p2_hand,
            #     contested_scores    =    contested_points,
            #     score_card_left     =    score_card_left
            # )
            # pid = proxy.get_next_agent()
            # print(f"{player2}, play a card out of {env.player2_hand}")
            # move2 = await player2.step(
            #     state               =    state,
            #     opponent_hand       =    reserved_p1_hand,
            #     contested_scores    =    contested_points,
            #     score_card_left     =    score_card_left
            # )
            # pid = proxy.get_next_agent()

            # await player1.observe_round(
            #     contested_points    =   contested_points,
            #     your_card           =   move1,
            #     opponent_card       =   move2,
            #     round_id            =   round_id
            # )
            # pid = proxy.get_next_agent()
            # print("Next player: ", pid)
            # await player2.observe_round(
            #     contested_points    =   contested_points,
            #     your_card           =   move2,
            #     opponent_card       =   move1,
            #     round_id            =   round_id
            # )
            # pid = proxy.get_next_agent()
            # print("Next player: ", pid)

            player_cards.append(move1)
            opponent_cards.append(move2)

            (done, score_card, contested_points) = env.play_cards(int(move1), int(move2))
            print(f"{player1} played {move1}, {player2} played {move2}")

        print(f"Final score: {player1}: {env.player1_score}, {player2}: {env.player2_score}")
        print("Thanks for playing GOPS!")

        finish_reason = SampleStatus.COMPLETED

        # return TaskSampleExecutionResult(status=finish_reason, result={
        #     "player1_score": int(env.player1_score),
        #     "player2_score": int(env.player2_score),
        #     "history of player 1": proxy.history[0],
        #     "history of player 2": proxy.history[1],
        # })
        return TaskSampleExecutionResult(status=finish_reason, result={
            "player1": self.agent_list[0],
            "player2": self.agent_list[1],
            "player1_score": int(env.player1_score),
            "player2_score": int(env.player2_score),
            "history of player 1": proxy.history[0],
            "history of player 2": proxy.history[1],
        })