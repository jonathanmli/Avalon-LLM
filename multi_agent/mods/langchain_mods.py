from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatAnthropic
from src.server.task import Session

from typing import List, Union, Dict
from src.typings import (
    AgentOutput,
    ChatHistoryItem,
)

class LangchainSession(Session):
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.1, api_key=None, **configs):
        super().__init__(**configs)
        if "gpt" in model_name:
            import os
            if api_key is None:
                try:
                    key = os.environ.get("OPENAI_API_KEY")
                except:
                    raise RuntimeError("OPENAI_API_KEY is not specified and not found in env variables")
            else:
                key = api_key

            self.langchain_model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=key
            )
        elif "claude" in model_name:
            import os
            if api_key is None:
                try:
                    key = os.environ.get("CLAUDE_API_KEY")
                except:
                    raise RuntimeError("CLAUDE_API_KEY is not specified and not found in env variables")
            else:
                key = api_key

            self.langchain_model = ChatAnthropic(
                model_name=model_name,
                temperature=temperature,
                anthropic_api_key=key
            )
        else:
            raise NotImplementedError(f"Model {model_name} not supported")

        
    def convert_history(self, history: List[ChatHistoryItem]) -> List[Union[HumanMessage, AIMessage]]:
        converted_history = []
        # if self.system_prompt:
        #     converted_history.append(SystemMessage(content=self.system_prompt))
        for item in history:
            if item.role == "user":
                converted_history.append(HumanMessage(content=item.content))
            elif item.role == "agent":
                converted_history.append(AIMessage(content=item.content))
            else:
                raise NotImplementedError(f"Role {item.role} not supported")
        return converted_history

    async def action(self, input: Dict=None, **kwargs) -> AgentOutput:
        if input is not None:
            self.inject(input)
        converted_history = self.convert_history(self.filter_messages(self.history))
        agent_response = await self.langchain_model.ainvoke(converted_history)
        self.history.append(
            ChatHistoryItem(
                role="agent", content=agent_response.content
            )
        )
        return agent_response.content