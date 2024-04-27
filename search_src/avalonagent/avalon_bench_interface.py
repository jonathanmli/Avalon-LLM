from AvalonBench import LLMAgentWithDiscussion # this part is not correct yet
from Avalon.baseline_models_Avalon import AvalonState

class SearchlightLLMAgentWithDiscussion(LLMAgentWithDiscussion):
    
    def __init__(self):
        super().__init__()
        # we want to keep track of the current game state here somehow. this should include both the current action history and dialogue history (or summary of dialogue history)
        # we will use modules from the dialogue_discrimination.py and dialogue_generator.py to update the game state and generate dialogue
        # and some action intent prediction module to predict the action intent of the agent
        raise NotImplementedError

    async def team_discussion(self, team_size, team_leader_id, mission_id):
        """
        Team discussion
        """
        raise NotImplementedError
    
    async def propose_team(self, team_size, mission_id):
        """
        Propose Team
        """
        raise NotImplementedError
    
    async def vote_on_team(self, team, mission_id):
        """
        Vote on team
        """
        raise NotImplementedError

    def convert_to_avalon_state(self, env) -> AvalonState:
        '''
        Converts the current Avalonbench game state to an AvalonState object
        '''
        return AvalonState.init_from_env(env)
    
    def get_action_intent(self):
        '''
        Returns the action intent of the agent
        '''
        state = self.convert_to_avalon_state()
        
        raise NotImplementedError
    
    