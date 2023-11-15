from copy import deepcopy
from typing import Dict, Union
from src.server.task import Session

class FakeSession:
    history: list=[]    # Fake history

    async def action(self, input: Dict):
        pass

    def inject(self, input: Dict):
        pass

class SessionWrapper:
    def __init__(self, session: Union[Session, FakeSession]):
        self.session = session