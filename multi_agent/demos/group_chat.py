# %%
import os
import sys
sys.path.append('../../')

# %%
from multi_agent.mods import LangchainSession
from multi_agent.proxy import MultiAgentProxy
from multi_agent.session_wrapper import SessionWrapper
from multi_agent.agents import ChatAgent
import asyncio

# %%
async def main():
    key = ""
    session = LangchainSession(api_key=key, temperature=0.7) # the LangchainSession can be replaced with a normal Session
    num_agents = 3
    proxy = MultiAgentProxy(session, num_agents)
    wrapper1 = SessionWrapper(session, proxy)
    wrapper2 = SessionWrapper(session, proxy)

    proxy.clean_buffer()
    proxy.clean_history()
    agent1 = ChatAgent(0, wrapper1, "You are Player 1. When generating responses, please directly output what you want to say. You don't need to output `Player x says:`")
    agent2 = ChatAgent(1, wrapper1, "You are Player 2. When generating responses, please directly output what you want to say. You don't need to output `Player x says:`")
    agent3 = ChatAgent(2, wrapper2, "You are Player 3. When generating responses, please directly output what you want to say. You don't need to output `Player x says:`")

    # Initialize the group chat with send()
    await agent1.send("all", "You will be talking to two other agents on the topic of jazz music. Say hi!", max_rounds=0)

    # Replying the message by calling reply()
    await agent2.reply()

    await agent3.reply()

if __name__ == "__main__":
    asyncio.run(main())