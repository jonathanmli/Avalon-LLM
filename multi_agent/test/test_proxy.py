from multi_agent.typings import FakeSession, Proxy
from multi_agent.proxy import MultiAgentProxy
from multi_agent.session_wrapper import SessionWrapper
import asyncio

async def test_broadcast(wrapped_session: SessionWrapper):
    await wrapped_session.action({
        'role': 'user',
        'content': ''
    },
    receiver="all")

async def test_normal_action(wrapped_session: SessionWrapper):
    await wrapped_session.action({
        'role': 'user',
        'content': ''
    })

async def test_single_conversation(wrapped_session: SessionWrapper):
    await wrapped_session.action({
        'role': 'user',
        'content': ''
    },
    receiver=3)

async def test_single_conversation_same_agent(wrapped_session: SessionWrapper):
    """
        This is a test to see if the proxy can handle message sent to the agent itself
    """
    await wrapped_session.action({
        'role': 'user',
        'content': ''
    },
    receiver=1)

async def test_require_reply(wrapped_session: SessionWrapper):
    try:
        await wrapped_session.action({
            'role': 'user',
            'content': ''
        },
        require_reply=True)
    except:
        print("passed")

def test_inject(wrapped_session: SessionWrapper):
    wrapped_session.inject({
        'role': 'user',
        'content': ''
    })

async def main():
    session = FakeSession()
    proxy = MultiAgentProxy(session, 5)
    proxy.set_current_agent(1)
    wrapped_session = SessionWrapper(session, proxy)

    await test_normal_action(wrapped_session)
    await test_single_conversation_same_agent(wrapped_session)
    await test_require_reply(wrapped_session)
    test_inject(wrapped_session)

if __name__ == "__main__":
    asyncio.run(main())


# python -m multi_agent.test.test_proxy