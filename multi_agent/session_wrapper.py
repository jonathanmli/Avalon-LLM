import queue
from copy import deepcopy
from typing import Dict, Union
from src.server.task import Session
from multi_agent.mods import LangchainSession
from src.typings import SampleStatus
from src.typings import AgentContextLimitException
from src.utils import ColorMessage

from multi_agent.typings import FakeSession, Proxy

class FakeSession:
    history: list=[]    # Fake history

    async def action(self, input: Dict):
        pass

    def inject(self, input: Dict):
        pass

class SessionWrapper:
    def __init__(self, session: Union[Session, FakeSession, LangchainSession], proxy: Proxy):
        self.session = session
        self.proxy = proxy
        self.decorate_method('action')
        self.decorate_method('inject')

    def decorate_method(self, method_name):
        # Get the method
        method = getattr(self, method_name)

        # Decorate and replace the method
        setattr(self, method_name, self.proxy.method_wrapper(method))

    def get_history(self):
        return self.session.history

    def overwrite_history(self, history: list):
        self.session.history = deepcopy(history)

    def inject(self, input: Dict, **kwargs):
        agent_id = kwargs.pop("agent_id", None)
        if agent_id is not None:
            print("Injecting: ", agent_id)
            self.proxy.set_current_agent(agent_id)
        self.session.inject(input)

    async def action(self, input: Dict=None, **kwargs):
        if input is not None:
            self.inject(input, **kwargs)
        print(self.proxy.session.history)
        # self.balance_history()
        response = await self.session.action()

        return response
    
    async def reply_all_messages(self, reply_agent: int):
        # self.proxy.set_current_agent(reply_agent)
        while not self.proxy.message_buffer[reply_agent].empty():
            message, max_rounds, sender = self.proxy.message_buffer[reply_agent].get()
            await self.proxy.generate_reply(
                message     =   message,
                max_rounds  =   max_rounds,
                sender      =   reply_agent,
                receiver    =   sender
            )

    async def reply_single_message(self, reply_agent: int, receiver: int):
        """
            Reply to a single message from a specific agent. 
            Only the first message from that agent will be replied
        """
        self.proxy.set_current_agent(reply_agent)
        if self.proxy.message_buffer[reply_agent].has_agent_id(receiver):
            message, max_rounds, sender = self.proxy.message_buffer[reply_agent].get_by_agent_id(receiver)
            print(max_rounds)
            await self.proxy.generate_reply(
                message     =   message,
                max_rounds  =   max_rounds,
                sender      =   reply_agent,
                receiver    =   sender
            )

    async def reply_messages_by_agent(self, reply_agent: int, receiver: int):
        """
            Reply all the messages from a specific agent. 
        """
        self.proxy.set_current_agent(reply_agent)
        while self.proxy.message_buffer[reply_agent].has_agent_id(receiver):
            message, max_rounds, sender = self.proxy.message_buffer[reply_agent].get_by_agent_id(receiver)
            await self.proxy.generate_reply(
                message     =   message,
                max_rounds  =   max_rounds,
                sender      =   reply_agent,
                receiver    =   sender
            )