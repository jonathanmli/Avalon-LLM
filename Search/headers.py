class State:
    '''
    Abstract class for a state

    state_type: 'max', 'min', 'random', 'player'
    'max': max value node, for the player
    'min': min value node, for the opponent trying to minimize the value
    'random': random value node, for the environment
    'player': player node, for other players with different goals
    '''
    STATE_TYPES = ['max', 'min', 'random', 'player']

    

    def __init__(self, id, state_type, notes = None):
        '''
        Args:
            id: id of the state, should be unique, usually the name of the state
            state_type: type of the state
            notes: any notes about the state
        '''
        self.id = id
        self.state_type = state_type # 'max', 'min', 'random', 'player'
        if state_type not in self.STATE_TYPES:
            raise ValueError(f"state_type must be one of {self.STATE_TYPES}")
        self.notes = notes

    def __repr__(self):
        return f"State({self.id}, {self.state_type}, {self.notes})"
    
    def __str__(self):
        return f"State({self.id}, {self.state_type}, {self.notes})"
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

class ForwardPredictor():
    '''
    Abstract class for a forward dynamics predictor
    '''

    def __init__(self):
        pass
    
    def predict(self, state: State, action, next_states):
        '''
        Predicts the probabilities over next states given the current state and action

        Args:
            state: current state
            action: action to take
            next_states: list of next states

        Returns:
            probs: list of probabilities over next states
        '''
        raise NotImplementedError
    
class ForwardEnumerator():
    '''
    Abstract class for a forward dynamics enumerator
    '''

    def __init__(self):
        pass

    def enumerate(self, state: State, action):
        '''
        Enumerates the possible next states given the current state and action

        Args:
            state: current state
            action: action to take

        Returns:
            next_states: list of next states
        '''
        raise NotImplementedError
    
class ActionPredictor():
    '''
    Abstract class for an action predictor
    '''

    def __init__(self):
        pass
    
    def predict(self, state: State, actions):
        '''
        Predicts the advantage of each action given the current state

        Args:
            state: current state
            actions: list of actions

        Returns:
            advantage: list of relative advantages of each action
        '''
        raise NotImplementedError
    
class ActionEnumerator():
    '''
    Abstract class for an action enumerator
    '''

    def __init__(self):
        pass

    def enumerate(self, state: State):
        '''
        Enumerates the possible actions given the current state 

        Args:
            state: current state

        Returns:
            actions: list of actions
        '''
        raise NotImplementedError

class RandomStatePredictor():
    '''
    Abstract class for a random dynamics predictor
    '''

    def __init__(self):
        pass
    
    def predict(self, state: State, next_states):
        '''
        Predicts the probabilities over next states given the current state and action

        Args:
            state: current state
            next_states: list of next states

        Returns:
            probs: list of probabilities over next states
        '''
        raise NotImplementedError
    
class RandomStateEnumerator():
    '''
    Abstract class for a random dynamics enumerator
    '''

    def __init__(self):
        pass

    def enumerate(self, state: State, action):
        '''
        Enumerates the possible next states given the current state and action

        Args:
            state: current state

        Returns:
            next_states: list of next states
        '''
        raise NotImplementedError

class OpponentActionPredictor():
    '''
    Abstract class for an opponent action predictor
    '''

    def __init__(self):
        pass
    
    def predict(self, state: State, actions):
        '''
        Predicts the advantage of each action given the current state

        Args:
            state: current state
            actions: list of actions

        Returns:
            advantage: list of relative advantages of each action
        '''
        raise NotImplementedError
    
class OpponentActionEnumerator():
    '''
    Abstract class for an opponent action enumerator
    '''

    def __init__(self):
        pass

    def enumerate(self, state: State):
        '''
        Enumerates the possible actions given the current state 

        Args:
            state: current state

        Returns:
            actions: list of actions
        '''
        raise NotImplementedError
    
class PolicyPredictor():
    '''
    Abstract class for a policy predictor
    '''

    def __init__(self):
        pass
    
    def predict(self, state: State, actions):
        '''
        Predicts the probabilities over actions given the current state

        Args:
            state: current state
            actions: list of actions

        Returns:
            probs: list of probabilities over actions
        '''
        raise NotImplementedError
    
class PolicyEnumerator():
    '''
    Abstract class for a policy enumerator
    '''

    def __init__(self):
        pass

    def enumerate(self, state: State):
        '''
        Enumerates the possible actions given the current state 

        Args:
            state: current state

        Returns:
            actions: list of actions
        '''
        raise NotImplementedError
    
class ValueHeuristic():
    '''
    Abstract class for a heuristic
    '''

    def __init__(self):
        pass
    
    def evaluate(self, state: State):
        '''
        Evaluates the state

        Args:
            state: current state

        Returns:
            value: value of the state
        '''
        raise NotImplementedError
    
class QHeuristic():
    '''
    Abstract class for a heuristic
    '''

    def __init__(self):
        pass
    
    def evaluate(self, state: State, action):
        '''
        Evaluates the state

        Args:
            state: current state
            action: action to take

        Returns:
            value: value of the state
        '''
        raise NotImplementedError


    
