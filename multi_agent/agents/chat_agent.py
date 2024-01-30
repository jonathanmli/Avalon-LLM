from multi_agent.session_wrapper import SessionWrapper
from multi_agent.mods import LangchainSession
from langchain.schema import SystemMessage

class ChatAgent:
    def __init__(self, id: int, session: SessionWrapper, sys_prompt: str=None):
        self.id = id
        self.session = session
        if sys_prompt:
            self.initialize_system_prompt(sys_prompt)

    def initialize_system_prompt(self, sys_prompt: str):
        if isinstance(self.session, LangchainSession):
            self.session.inject({
                "role": "user",
                "content": SystemMessage(content=sys_prompt)
            }, agent_id=self.id)
        else:
            self.session.inject({
                "role": "user",
                "content": sys_prompt
            }, agent_id=self.id)

    async def send(self, receiver: int, message: str, max_rounds: int=0):
        self.session.inject({
            'role': 'user',
            'content': message
        }, agent_id=self.id)
        await self.session.action(
            sender=self.id,
            receiver=receiver,
            max_rounds=max_rounds
        )

    async def reply_all(self):
        await self.session.reply_all_messages(self.id)

    async def reply(self, max_rounds: int=0):
        await self.session.action(
            sender=self.id,
            receiver="all",
            max_rounds=max_rounds,
            require_reply=True
        )

    async def reply_single(self, receiver: int):
        await self.session.reply_single_message(
            reply_agent=self.id,
            receiver=receiver
        )