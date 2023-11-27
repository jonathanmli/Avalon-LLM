from typing import Dict, List
from src.server.task import Session
from .typings import FakeSession, Proxy

from copy import deepcopy
import asyncio
import functools

class MultiAgentProxy(Proxy):
    """The proxy class that wraps around the methods of the session class, maintains history of each agent, and controls the order of the agents.

    Args:
        session (Session): The session class that the proxy wraps around.
        num_agents (int): The number of agents that will be using the proxy.

    Methods:
        method_wrapper: Wraps around the given (session) method.

        get_next_agent: Returns the id of the next agent.
        set_current_agent: Sets the current agent to the given id.
    """
    def __init__(self, session: Session, num_agents: int):
        self.session = session
        self.num_agents = num_agents
        self.current_agent = 0
        self.history = [[] for _ in range(num_agents)]
        # self.session_list = session_list
        # self.session_list[0] = session
        

    def update_history(player_id: int, history: Dict):
        pass

    def initialize_sessions(self, session_list: List):
        self.session_list = session_list

    def method_wrapper(self, method):
        @functools.wraps(method)
        def sync_wrapper(*args, **kwargs):
            print(f"Before calling {method.__name__}")
            print("NUM AGENTS: ", self.session)
            print("AGENT LIST: ", self.session_list)
            self.session.history = deepcopy(self.history[self.current_agent])
            self.session_list[self.current_agent].session = self.session
            result = method(*args, **kwargs)
            print(self.session.history)
            # self.session_list[self.current_agent].session = None
            self.history[self.current_agent] = deepcopy(self.session.history)
            # print(self.history[self.current_agent])
            print(f"After calling {method.__name__}")
            return result
        
        @functools.wraps(method)
        async def async_wrapper(*args, **kwargs):
            print(f"Before calling {method.__name__}")
            print("NUM AGENTS: ", self.session)
            print("AGENT LIST: ", self.session_list)
            self.session.history = deepcopy(self.history[self.current_agent])
            self.session_list[self.current_agent].session = self.session
            result = await method(*args, **kwargs)
            print(self.session.history)
            # self.session_list[self.current_agent].session = None
            self.history[self.current_agent] = deepcopy(self.session.history)
            # print(self.history[self.current_agent])
            print(f"After calling {method.__name__}")
            return result
        
        if asyncio.iscoroutinefunction(method):
            return async_wrapper
        else:
            return sync_wrapper

    def wrap_specific_methods(self, *method_names):
        def decorator(target_cls):
            for method_name in method_names:
                if hasattr(target_cls, method_name):
                    original_method = getattr(target_cls, method_name)
                    if callable(original_method):
                        # Need to pass 'self' explicitly to method_wrapper
                        setattr(target_cls, method_name, self.method_wrapper(original_method))
            return target_cls
        return decorator
    
    # def get_sessions(self):
    #     return self.session_list
    
    def get_next_agent(self) -> int:
        self.current_agent = (self.current_agent + 1) % self.num_agents
        return self.current_agent
    
    def set_current_agent(self, agent_id: int) -> int:
        self.current_agent = agent_id
        return self.current_agent