from typing import Dict

class FakeSession:
    history: list=[]    # Fake history

    async def action(self, input: Dict):
        pass

    def inject(self, input: Dict):
        pass

class Proxy:
    def __init__(self):
        pass