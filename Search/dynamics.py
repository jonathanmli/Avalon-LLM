from headers import *
from prompts import *
import re
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import List, float

# Parse helper funcs
def parse_bracketed_list(string: str) -> List[str]:
    pattern = r'\[([^\]]+)\]'

    matches = re.findall(pattern, string)

    items = [item.strip() for item in matches[0].split(',')] if matches else []

    return items

def parse_int_value(string: str) -> int:
    pattern = r'\b\d+\b'

    integers = [int(num) for num in re.findall(pattern, string)]

    return integers[-1] # should be designed in the prompt that the last num is the value

def parse_prob_value(string: str) -> float:
    pattern = r'\b\d+\.\d+|\b\d+|\.\d+\b'

    floats = [float(num) for num in re.findall(pattern, string)]

    return floats[-1]


class GPT35ForwardEnumerator(ForwardEnumerator):
    '''
    Forward dynamics enumerator for GPT-3.5
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model  #   The model here refers to the Session in AgentBench

    def enumerate(self, state: State, action) -> List[State]:
        '''
        Enumerates the possible next states given the current state and action

        Args:
            state: current state
            action: action to take

        Returns:
            next_states: list of next states
        '''
        # Prepare input
        input_prompt = "Current State: {state}\nCurrent Action: {action}\n".format(state=state, action=action)
        input_prompt += FORWARD_ENUMERATOR_PROMPT

        # Call the model
        output = self.model.single_action(input_prompt)

        # Parse the output
        verbal_states = parse_bracketed_list(output)
        next_states = [State(id=id, state_type='max') for id in verbal_states]

        return next_states
    
class GPT35ForwardPredictor(ForwardPredictor):
    '''
    Forward dynamics predictor for GPT-3.5
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model

    def predict(self, state: State, action, next_states) -> List[float]:
        '''
        Predicts the next state given the current state and action

        Args:
            state: current state
            action: action to take
            next_states: next states obtained from GPT35ForwardEnumerator
        Returns:
            probs: probabilities for next states
        '''
        # Prepare input
        input_prompt = "Current State: {state}\nCurrent Action: {action}\nNext States: {next_states}".format(
            state       =   state,
            action      =   action,
            next_states =   next_states
        )
        input_prompt += FORWARD_PREDICTOR_PROMPT

        # Call the model
        output = self.model.single_action(input_prompt)

        # Parse the output
        verbal_probs = parse_bracketed_list(output)
        probs = [float(prob) for prob in verbal_probs]

        return probs
    
class GPT35RandomStateEnumerator(RandomStateEnumerator):
    '''
    Hidden state enumerator for GPT-3.5
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model

    def enumerate(self, state: State, action) -> List[State]:
        '''
        Enumerates the possible next hidden states given the current state and action

        Args:
            state: current state
            action: action to take

        Returns:
            next_hidden_states: list of next hidden states
        '''
        # Prepare input
        input_prompt = "Current State: {state}\nCurrent Action: {action}\n".format(state=state, action=action)
        input_prompt += HIDDEN_STATE_ENUMERATOR_PROMPT

        # Call the model
        output = self.model.single_action(input_prompt)

        # Parse the output
        verbal_states = parse_bracketed_list(output)
        next_hidden_states = [State(id=id, state_type='random') for id in verbal_states]

        return next_hidden_states
    
class GPT35RandomStatePredictor(RandomStatePredictor):
    '''
    Hidden state predictor for GPT-3.5
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model

    def predict(self, state: State, action) -> List[float]:
        '''
        Predicts the probabilities over next hidden states given the current state and action

        Args:
            state: current state
            action: action to take

        Returns:
            hidden_probs: list of probabilities over hidden states
        '''
        # Prepare input
        input_prompt = "Current State: {state}\nCurrent Action: {action}\n".format(state=state, action=action)
        input_prompt += HIDDEN_STATE_PREDICTOR_PROMPT

        # Call the model
        output = self.model.single_action(input_prompt)

        # Parse the output
        verbal_probs = parse_bracketed_list(output)
        hidden_probs = [float(prob) for prob in verbal_probs]

        return hidden_probs
    
class GPT35OpponentActionEnumerator(OpponentActionEnumerator):
    '''
    Opponent action enumerator for GPT-3.5
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model

    def enumerate(self, state: State, action) -> List[int]:
        '''
        Enumerates the possible opponent actions given the current state and action

        Args:
            state: current state
            action: action to take

        Returns:
            opponent_actions: list of opponent actions; for goopspiel, the actions are the card to play
        '''
        # Prepare input
        input_prompt = "Current State: {state}\nCurrent Action: {action}\n".format(state=state, action=action)
        input_prompt += OPPONENT_ACTION_ENUMERATOR_PROMPT

        # Call the model
        output = self.model.single_action(input_prompt)

        # Parse the output
        verbal_actions = parse_bracketed_list(output)
        opponent_actions = [int(action) for action in verbal_actions]

        return opponent_actions
    
class GPT35OpponentActionPredictor(OpponentActionPredictor):
    '''
    Opponent action predictor for GPT-3.5
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
            tokenizer: GPT-3.5 tokenizer
            device: torch device
        '''
        self.model = model

    def predict(self, state: State, actions) -> List[float]:
        '''
        Predicts the advantage of each opponent action given the current state and action

        Args:
            state: current state
            actions: actions to take

        Returns:
            advantage: list of relative advantages of each opponent action (probs for current implementation)
        '''
        # Prepare input
        input_prompt = "Current State: {state}\nCurrent Action: {action}\n".format(state=state, action=action)
        input_prompt += OPPONENT_ACTION_PREDICTOR_PROMPT

        # Call the model
        output = self.model.single_action(input_prompt)

        # Parse the output
        verbal_output = parse_bracketed_list(output)
        advantages = [float(advantage) for advantage in verbal_output]

        return advantages
    
class GPT35ValueHeuristic(ValueHeuristic):
    '''
    Value heuristic for GPT-3.5
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model

    def predict(self, state: State) -> int:
        '''
        Predicts the value of the state

        Args:
            state: current state

        Returns:
            value: value of the state
        '''
        # Prepare input
        prob_prompt = "Current State: {state}\n".format(state=state)
        prob_prompt += VALUE_PREDICTOR_PROMPTS[0]
        value_prompt = "Current State: {state}\n".format(state=state)
        value_prompt += VALUE_PREDICTOR_PROMPTS[1]

        # Call the model
        prob_output = self.model.single_action(prob_prompt)
        value_output = self.model.single_action(value_prompt)

        # Parse the output
        prob_value = parse_prob_value(prob_output)
        value = parse_int_value(value_output)

        return value
    
class GPT35ActionEnumerator(ActionEnumerator):
    '''
    Action enumerator for GPT-3.5
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model

    def enumerate(self, state: State) -> List[int]:
        '''
        Enumerates the possible actions given the current state

        Args:
            state: current state

        Returns:
            actions: list of actions
        '''
        # Prepare input
        input_prompt = "Current State: {state}\n".format(state=state)
        input_prompt += ACTION_ENUMERATOR_PROMPT

        # Call the model
        output = self.model.single_action(input_prompt)

        # Parse the output
        verbal_actions = parse_bracketed_list(output)
        actions = [int(action) for action in verbal_actions]

        return actions
    
if __name__ == "__main__":
    class GPT35:
        def __init__(self):
            import os
            key = os.environ.get("OPENAI_API_KEY")
            self.model = ChatOpenAI(temperature=0.1, openai_api_key=key)
        def single_action(self, input_prompt: str):
            input_prompt = HumanMessage(content=input_prompt)
            output = self.model(input_prompt).content

            return output

    model = GPT35()

    # Instantiate the dynamics
    actionenumerator = GPT35ActionEnumerator(model)
    valueheuristic = GPT35ValueHeuristic(model)
    opponentactionpredictor = GPT35OpponentActionPredictor(model)
    opponentactionenumerator = GPT35OpponentActionEnumerator(model)
    hiddenstatepredictor = GPT35RandomStatePredictor(model)
    hiddenstateenumerator = GPT35RandomStateEnumerator(model)
    forwardpredictor = GPT35ForwardPredictor(model)
    forwardenumerator = GPT35ForwardEnumerator(model)

