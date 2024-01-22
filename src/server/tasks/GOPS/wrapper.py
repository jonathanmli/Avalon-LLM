from copy import deepcopy
from typing import Dict, Union
from src.server.task import Session
from src.typings import SampleStatus
from src.typings import AgentContextLimitException
import re

from multi_agent.typings import FakeSession, Proxy
from multi_agent.session_wrapper import SessionWrapper

class GOPSSessionWrapper(SessionWrapper):
    def __init__(self, session: Union[Session, FakeSession], proxy: Proxy):
        self.session = session
        self.proxy = proxy
        self.decorate_method('action')
        self.decorate_method('inject')

    def decorate_method(self, method_name):
        # Get the method
        method = getattr(self, method_name)

        # Decorate and replace the method
        setattr(self, method_name, self.proxy.method_wrapper(method))

    async def action(self, input: Dict):
        if isinstance(self.session, Session):
            print("SESSION")
            self.inject({
                "role": input['role'],
                "content": input['content']
            })
            response = await self.session.action()
            if response.status == SampleStatus.AGENT_CONTEXT_LIMIT:
                raise AgentContextLimitException()
            if response.content is None:
                raise RuntimeError("Response content is None.")
            print(response.content)
            filtered_response = response.content.split("Decision:")[-1]
            if input['mode'] != "system":
                matches = re.findall(r'\d+', filtered_response)
                print("List of possible cards: ", list(matches))
                return list(matches)
            else:
                return None
        elif isinstance(self.session, FakeSession):
            print("FAKE SESSION")
            return input.pop('naive_result', None)
    
    def inject(self, input: Dict):
        print("INJECT")
        return self.session.inject(input)