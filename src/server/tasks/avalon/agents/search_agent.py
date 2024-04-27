from .llm_with_discussion import LLMAgentWithDiscussion # this part is not correct yet
from ..engine import AvalonBasicConfig
from ..wrapper import AvalonSessionWrapper, Session
from search_src.Avalon.baseline_models_Avalon import AvalonState

class SearchlightLLMAgentWithDiscussion(LLMAgentWithDiscussion):
    
    def __init__(self, name: str, num_players: int, id: int, role: int, role_name: str, config:AvalonBasicConfig, session: AvalonSessionWrapper=None, side=None, seed=None, **kwargs):
        # we want to keep track of the current game state here somehow. this should include both the current action history and dialogue history (or summary of dialogue history)
        # we will use modules from the dialogue_discrimination.py and dialogue_generator.py to update the game state and generate dialogue
        # and some action intent prediction module to predict the action intent of the agent
        super().__init__(
            name=name,
            num_players=num_players,
            id=id,
            role=role,
            role_name=role_name,
            config=config,
            session=session,
            side=side,
            seed=seed,
            **kwargs
        )

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

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
    