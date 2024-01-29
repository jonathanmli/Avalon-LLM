# %%
import os
import sys
sys.path.append('../../')

# %%
from multi_agent.mods import LangchainSession
from multi_agent.proxy import MultiAgentProxy
from multi_agent.session_wrapper import SessionWrapper
import asyncio

# %%
async def main():
    key = ""
    session = LangchainSession(api_key=key) # the LangchainSession can be replaced with a normal Session
    num_agents = 2
    proxy = MultiAgentProxy(session, num_agents)
    wrapper = SessionWrapper(session, proxy)

    from multi_agent.agents import ChatAgent

    agent1 = ChatAgent(0, wrapper)
    agent2 = ChatAgent(1, wrapper)

    proxy.clean_buffer()
    proxy.clean_history()
    await agent1.send(1, "You will be talking to another agent. Say hi!", max_rounds=10)
    end = False
    idx = 0
    agent_list = [agent1, agent2]
    while not end:
        end = True
        for idx in range(0, 2):
            agent = agent_list[idx]
            if not proxy.message_buffer[agent.id].empty():
                print(f"test {idx}")
                end = False
                await agent.reply_single((idx + 1) % 2)

    await agent2.reply_all()

    await agent1.reply_all()

if __name__ == "__main__":
    asyncio.run(main())