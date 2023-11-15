import sys
import json
from copy import deepcopy
from typing import List, Tuple, Dict, Any

from src.server.task import Task, Session
from src.typings import TaskSampleExecutionResult, TaskOutput, SampleIndex, AgentOutputStatus, SampleStatus
from src.utils import ColorMessage

from .engine import *
from .agents.naive import NaiveGOPSAgent

from .wrapper import FakeSession, SessionWrapper

from src.typings import AgentContextLimitException

AGENT_FINDER = {
    'naive': NaiveGOPSAgent,
}

class GOPSBench(Task):
    def __init__(self, num_games, num_turns, agent_list, **configs):
        super().__init__(**configs)

        self.num_games = num_games
        self.num_turns = num_turns
        self.agent_list = agent_list

        self.data = [0 for _ in range(num_games)]


    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        print("testtesttest")
        outputs = [None for _ in range(len(self.data))]
        for result in results:
            outputs[result.index] = result.result

        player1_wins = 0
        player2_wins = 0
        ties = 0
        for output in outputs:
            if output["player1_score"] > output["player2_score"]:
                player1_wins += 1
            elif output["player1_score"] < output["player2_score"]:
                player2_wins += 1
            else:
                ties += 1

        return {
            "winrate of player 1": player1_wins / len(self.data),
            "winrate of player 2": player2_wins / len(self.data),
            "tie rate": ties / len(self.data)
        }

    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.data)))

    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        assert isinstance(index, int), "Index must be an integer"
        config = GOPSConfig(self.num_turns)
        env = GOPSEnvironment(config)

        sessions = [SessionWrapper(session), SessionWrapper(FakeSession())]

        (done, score_card, contested_points) = env.reset()

        # Initialize players
        # Should let LLM be player 1 by default
        player1 = AGENT_FINDER[self.agent_list[0]](
            id      =   0,
            hand    =   deepcopy(env.player1_hand),
            session =   sessions[0]
        )
        player2 = AGENT_FINDER[self.agent_list[1]](
            id      =   1,
            hand    =   deepcopy(env.player2_hand),
            session =   sessions[1]
        )

        print(f"Welcome {player1} and {player2} to GOPS!")
        while not done:
            print(f"Current score: {player1}: {env.player1_score}, {player2}: {env.player2_score}")
            print(f"Current contested points: {contested_points}, current contested score card: {score_card}")

            print(f"{player1}, play a card out of {env.player1_hand}")
            move1 = await player1.play_card()
            print(f"{player2}, play a card out of {env.player2_hand}")
            move2 = await player2.play_card()

            (done, score_card, contested_points) = env.play_cards(int(move1), int(move2))
            print(f"{player1} played {move1}, {player2} played {move2}")

        print(f"Final score: {player1}: {env.player1_score}, {player2}: {env.player2_score}")
        print("Thanks for playing GOPS!")

        finish_reason = SampleStatus.COMPLETED

        return TaskSampleExecutionResult(status=finish_reason, result={
            "player1_score": int(env.player1_score),
            "player2_score": int(env.player2_score)
        })