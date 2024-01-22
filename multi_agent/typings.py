from typing import Dict, Any, Tuple

class FakeSession:
    history: list=[]    # Fake history

    async def action(self, input: Dict):
        pass

    def inject(self, input: Dict):
        pass

class Proxy:
    def __init__(self):
        pass

class MessageBuffer:
    def __init__(self):
        self.buffer = []

    def put(self, elm: Tuple[Dict, int, int]):
        self.buffer.append(elm)

    def get(self):
        return self.buffer.pop(0)
    
    def has_agent_id(self, agent_id: int):
        for i in range(len(self.buffer)):
            if self.buffer[i][2] == agent_id:
                return True
        return False
    
    def get_by_agent_id(self, agent_id: int):
        for i in range(len(self.buffer)):
            if self.buffer[i][2] == agent_id:
                return self.buffer.pop(i)
            
        # If not found, raise error
        raise RuntimeError(f"Agent {agent_id} has no message in buffer.")

    def empty(self):
        return len(self.buffer) == 0
        