import sys
import json
from copy import deepcopy
from typing import List, Tuple, Dict, Any

from src.server.task import Task, Session
from src.typings import TaskSampleExecutionResult, TaskOutput, SampleIndex, AgentOutputStatus, SampleStatus
from src.utils import ColorMessage

from multi_agent.typings import FakeSession

from multi_agent.proxy import MultiAgentProxy

class MultiagentTest(Task):
    def __init__(self, **configs):
        super().__init__(**configs)


    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        outputs = [None for _ in range(len(self.data))]

        return {
            "test": "test"
        }

    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.data)))

    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        test = "This is a task for testing the multi-agent module"

        finish_reason = SampleStatus.COMPLETED

        return TaskSampleExecutionResult(status=finish_reason, result={
            "test": test
        })