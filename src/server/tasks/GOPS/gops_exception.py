from typing import Union
from src.typings import AgentBenchException

class GOPSException(AgentBenchException):
    def __init__(self, reason: str, detail: Union[str, None] = None) -> None:
        super().__init__()
        self.reason = reason
        self.detail = detail

    def __str__(self) -> str:
        if not self.detail:
            return "{CLASS_NAME}[{REASON}]".format(
                CLASS_NAME=self.__class__.__name__, REASON=self.reason
            )
        else:
            return "{CLASS_NAME}[{REASON}]: {DETAIL}".format(
                CLASS_NAME=self.__class__.__name__,
                REASON=self.reason,
                DETAIL=self.detail,
            )

class GOPSEnvException(GOPSException):
    def __init__(self, detail: Union[str, None] = None) -> None:
        super().__init__("Avalon Environment Exception", detail)

class GOPSAgentActionException(GOPSException):
    def __init__(self, detail: Union[str, None] = None) -> None:
        super().__init__("Invalid action (result) with retry", detail)