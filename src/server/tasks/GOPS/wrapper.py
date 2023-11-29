from copy import deepcopy
from typing import Dict, Union
from src.server.task import Session
from src.typings import SampleStatus
from src.typings import AgentContextLimitException
import re

from multi_agent.typings import FakeSession, Proxy

class SessionWrapper:
    def __init__(self, session: Union[Session, FakeSession], proxy: Proxy):
        self.session = session
        self.proxy = proxy
        self.decorate_method('action')
        self.decorate_method('inject')

    def balance_history(self):
        '''
            TODO: Make this function look better
        '''
        if len(self.session.history) % 2 != 0:
            self.inject({
                'role': 'user',
                'content': ''
            })

    def decorate_method(self, method_name):
        # Get the method
        method = getattr(self, method_name)

        # Decorate and replace the method
        setattr(self, method_name, self.proxy.method_wrapper(method))

    async def action(self, input: Dict):
        if isinstance(self.session, Session):
            print("SESSION")
            self.balance_history()
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
            if input['mode'] != "system":
                matches = re.findall(r'\d+', response.content)
                return int(matches[-1])
            else:
                return None
        elif isinstance(self.session, FakeSession):
            print("FAKE SESSION")
            return input.pop('naive_result', None)
    
    def inject(self, input: Dict):
        print("INJECT")
        return self.session.inject(input)