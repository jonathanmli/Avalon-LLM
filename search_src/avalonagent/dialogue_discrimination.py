from searchlight.utils import AbstractLogged

class DialogueDiscriminator(AbstractLogged):
    '''
    Class for dialogue discrimination. Dialogue discrimination is the process of converting the dialogue from the previous round and summary of discussions from rounds before that into a better representation (i.e. numerical beliefs). The dialogue discriminator is used to update the beliefs of the agent based on the input string.
    '''
    def __init__(self, llm_model):
        self.llm_model = llm_model # LLM model for dialogue discrimination
    
    def update_beliefs(self, history: str):
        '''
        Updates the beliefs based on the input string. The input string might include the dialogue from the previous round and summary of discussions from rounds before that.

        Args:
            input_str: input string to update the beliefs
        '''
        raise NotImplementedError

    def get_beliefs(self) -> dict:
        '''
        Returns the beliefs of the agent
        '''
        raise NotImplementedError
        
        