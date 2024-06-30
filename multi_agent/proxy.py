from typing import Dict, List, Union
from src.server.task import Session
from .typings import FakeSession, Proxy, MessageBuffer
from multi_agent.mods import LangchainSession

from copy import deepcopy
import asyncio
import functools
import warnings

class MultiAgentProxy(Proxy):
    """The proxy class that wraps around the methods of the session class, maintains history of each agent, and controls the order of the agents.

    Args:
        session (Union[Session, FakeSession]): The session class that the proxy wraps around. FakeSession is used for testing
        num_agents (int): The number of agents that will be using the proxy.
        conversation_template (str, optional): The template of the conversation. Defaults to "{agent_name} says:\n{message}".

    Methods:
        method_wrapper: Wraps around the given (session) method.
            sync_wrapper:
                For now, only session.inject()
            async_wrapper:
                For now, only session.action()

                The message passing feature is implemented here.The output of the wrapped methods will be passed to the receivers. \
                The message passing will be decided by params in kargs. Different circumstances include:
                    1. kargs['receiver'] is `None`: No message passing
                    2. kargs['receiver'] is "all" (`str`): broadcasting the message sent by the sender kargs['sender']
                    3. kargs['receiver'] is `int`: sending the message to the corresponding agent. The message will be appended \
                        to the corresponding agent history
                    4. kargs['receiver'] is `list` or `tuple`: sending the message to the corresponding agents. The message will be \
                        appended to the corresponding agent history

                We also support consistent conversation. It is decided by kargs['max_rounds'].
                    1. If kargs['max_rounds'] == 0, the receiver will not be asked to pass the message to the sender.
                    2. If kargs['max_rounds'] > 0, the receiver will be asked to pass the message to the sender. Each message passing \
                        counts as a round. The message passing will stop when the number of rounds reaches kargs['max_rounds'].
                    Or, you can pass require_reply=True to request a reply, which is equivalent to max_rounds == 1

        get_next_agent: Returns the id of the next agent.
        set_current_agent: Sets the current agent to the given id.
    """
    def __init__(self, session: Union[Session, FakeSession, LangchainSession], num_agents: int, conversation_template: str = "{agent_name} says:\n{message}"):
        self.session = session
        self.num_agents = num_agents
        self.conversation_template = conversation_template
        self.current_agent = 0
        self.history = [[] for _ in range(num_agents)]

        self.initialize_sessions([session for _ in range(num_agents)])
        self.get_agent_names()
        self.message_buffer = [MessageBuffer() for _ in range(num_agents)] # Buffer for messages, implemented as an iterable FIFO queue
        
    def balance_history(self):
        '''
            TODO: Make this function look better
        '''
        concatenated_content = ""
        # print(self.session.history)
        if len(self.session.history) % 2 != 1:
            for i in range(len(self.session.history)-1, -1, -1):
                if self.session.history[i].role == "agent":
                    break
                else:
                    concatenated_content = self.session.history.pop(i).content + "\n" + concatenated_content
                    # print("Popped: ", concatenated_content)
        if concatenated_content != "":
            self.session.inject({
                "role": "user",
                "content": concatenated_content
            })
        # print("After: ", self.session.history)


    def update_history(self, agent_id: int, history: Dict):
        self.history[agent_id] = deepcopy(history)


    def initialize_sessions(self, session_list: List):
        self.session_list = session_list

    
    def get_agent_names(self):
        self.agent_names = ['' for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.agent_names[i] = f"Player {i+1}"

    def reply_single_message(self, sender: int, reply_agent: int):
        pass


    async def reply_all_messages(self, reply_agent: int):
        pass


    def method_wrapper(self, method):
        @functools.wraps(method)
        def sync_wrapper(*args, **kwargs):
            self.session.history = deepcopy(self.history[self.current_agent])
            result = method(*args, **kwargs)
            self.balance_history()
            # print("Injecting")
            self.history[self.current_agent] = deepcopy(self.session.history)
            # print("INJECT Result: ", self.session.history)

            return result
        
        @functools.wraps(method)
        async def async_wrapper(*args, **kwargs):
            sender = kwargs.pop("sender", self.current_agent)
            receiver = kwargs.pop("receiver", None)
            max_rounds = kwargs.pop("max_rounds", 0)
            require_reply = kwargs.pop("require_reply", False)
            print("Action...")
            self.set_current_agent(sender)

            self.balance_history()

            # take action
            result = await method(*args, **kwargs)

            self.update_history(self.current_agent, self.session.history)

            if not isinstance(result, str):
                result = str(result)

            if max_rounds > 0:
                require_reply = True

            if require_reply:
                max_rounds = max(1, max_rounds)

            if max_rounds > 0 and receiver is None:
                raise RuntimeError(
                    f"max_rounds {max_rounds} is larger than 0, whereas the receiver {receiver} is not specified"
                )

            if receiver == None:
                pass
            elif receiver == "all":
                # print(f"Sender is {sender}")
                for idx in range(self.num_agents):
                    if idx == sender:
                        continue
                    self.set_current_agent(idx)
                    # print(f"Current {idx} history:\n{self.session.history}")
                    # TODO: support flexible message format
                    post_processed_message = self.conversation_template.format(
                        agent_name=self.agent_names[sender],
                        message=result
                    )
                    # TODO: this will directly call the session.inject, make sure you are aware of this
                    if max_rounds == 0:
                        self.session.inject({
                            "role": "user",
                            "content": post_processed_message
                        })
                    else:
                        self.message_buffer[self.current_agent].put(({
                            "role": "user",
                            "content": post_processed_message
                        }, 
                        max_rounds-1, sender))
                        
            elif isinstance(receiver, int):
                if receiver == sender:
                    warnings.warn("The sender {} and the receiver {} are the same, please make sure you are aware of this.".format(self.agent_names[sender], self.agent_names[receiver]))
                self.set_current_agent(receiver)
                if require_reply:
                    self.message_buffer[receiver].put(({
                        "role": "user",
                        "content": result
                    }, max_rounds-1, sender))
                else:
                    self.session.inject({
                        "role": "user",
                        "content": result
                    })
            elif isinstance(receiver, Union[List, tuple]):
                for idx in receiver:
                    if idx == sender:
                        continue
                    self.set_current_agent(idx)
                    if require_reply:
                        self.message_buffer[idx].put(({
                            "role": "user",
                            "content": result
                        }, max_rounds-1, sender))
                    else:
                        self.session.inject({
                            "role": "user",
                            "content": result
                        })
            else:
                raise NotImplementedError(f"Unsupported receiver type: {type(receiver)}")
            
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
    

    async def generate_reply(self, message: Dict, max_rounds: int, sender: int, receiver: int):
        self.set_current_agent(sender)
        self.session.history = deepcopy(self.history[self.current_agent])

        if max_rounds == 0:
            self.session.inject(message)
            self.balance_history()
        else:
            self.session.inject(message)
            self.balance_history()
            reply = await self.session.action()
            print(reply)
            self.message_buffer[receiver].put(({
                "role": "user",
                "content": reply,
            }, max_rounds-1, sender))
        self.history[self.current_agent] = deepcopy(self.session.history)


    def clean_buffer(self,):
        self.message_buffer = [MessageBuffer() for _ in range(self.num_agents)]


    def clean_history(self):
        self.history = [[] for _ in range(self.num_agents)]
        self.session.history = []

    
    def get_next_agent(self) -> int:
        next_agent_id = (self.current_agent + 1) % self.num_agents
        self.set_current_agent(next_agent_id)
        return self.current_agent
    
    
    def set_current_agent(self, agent_id: int) -> int:
        print("Setting Current Agent to ", agent_id)
        self.history[self.current_agent] = deepcopy(self.session.history)
        self.current_agent = agent_id
        self.session.history = deepcopy(self.history[self.current_agent])
        return self.current_agent