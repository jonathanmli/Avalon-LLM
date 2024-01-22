class State:
    '''
    Abstract class for a state
    '''
    
    def __init__(self, id, actors = None, done=False, reward=0.0, notes = None):
        '''
        Args:
            id: id of the state, should be unique, usually the name of the state
            state_type: type of the state
            notes: any notes about the state
            done: whether the state is done
            reward: reward of the state
            actors: actors that may take actions at the state

        Some conventions on actor names:
            -1: environment
            0: player 1, the main player who is trying to maximize reward
            1: player 2, usually the opponent player who is trying to minimize reward
            2+: other adaptive actors
        '''
        self.id = id
        self.notes = notes
        self.done = done
        self.reward = reward # TODO: rewards are associated with states and not actions at the moment
        if actors is None:
            actors = frozenset()
        self.actors = frozenset(actors)

    def is_done(self):
        '''
        Returns whether the state is done
        '''
        return self.done
    
    def get_reward(self):
        '''
        Returns the reward of the state
        '''
        return self.reward
    
    def copy(self):
        '''
        Returns a copy of the state
        '''
        return State(self.id, self.actors, self.done, self.reward, self.notes)

    def __repr__(self):
        return f"State({self.id}, {self.actors}, {self.done}, {self.reward}, {self.notes})"
    
    def __str__(self):
        return f"State({self.id}, {self.actors}, {self.done}, {self.reward}, {self.notes})"
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id
    
class ForwardTransitor():
    '''
    Abstract class for a forward dynamics transitor
    '''
    def __init__(self):
        pass

    def transition(self, state: State, actions: dict)->State:
        '''
        Transits to the next state given the current state and action

        Args:
            state: current state
            actions: actions taken by the actors

        Returns:
            next_state: next state
        '''
        raise NotImplementedError
    
class ActorEnumerator():
    '''
    Abstract class for an actor enumerator
    '''
    def __init__(self):
        pass

    def enumerate(self, state: State)->set:
        '''
        Enumerates the actors that may take actions at the state

        Args:
            state: current state

        Returns:
            actors: set of actors that may take actions at the state
        '''
        return set()
    
class ActionPredictor():
    '''
    Abstract class for an action predictor (policy predictor)
    '''

    def __init__(self):
        pass
    
    def predict(self, state: State, actions, actor) -> dict:
        '''
        Predicts the policy probabilities across actions given the current state and actor

        Args:
            state: current state
            actions: list of actions
            actor: actor to predict policy for

        Returns:
            probs: dictionary of actions to probabilities
        '''
        raise NotImplementedError
    
class ActionEnumerator():
    '''
    Abstract class for an action enumerator
    '''

    def __init__(self):
        pass

    def enumerate(self, state: State, actor) -> set:
        '''
        Enumerates the possible actions given the current state and actor

        Args:
            state: current state
            actor: actor to enumerate actions for

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
    
    def predict(self, state: State, actions)-> dict:
        '''
        Predicts the probabilities over actions given the current state

        Args:
            state: current state
            actions: list of actions

        Returns:
            probs: dictionary of probabilities over actions
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


    
