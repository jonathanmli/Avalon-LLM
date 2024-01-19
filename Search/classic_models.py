from Search.headers import *
import numpy as np

class RandomActionPredictor(ActionPredictor):

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
        probs = dict()
        for action in actions:
            probs[action] = 1/len(actions)
        return probs

class RandomRolloutValueHeuristic(ValueHeuristic):

    def __init__(self, actor_enumerator: ActorEnumerator, action_enumerator: ActionEnumerator,
                  forward_transitor: ForwardTransitor, action_predictor: ActionPredictor = None,
                  num_rollouts=100, random_state: np.random.RandomState=None):
        '''
        Args:
            actor_enumerator: actor enumerator
            action_enumerator: action enumerator
            action_predictor: action predictor
            forward_transitor: forward transitor
            num_rollouts: number of rollouts to perform
        '''
        super().__init__()
        self.actor_enumerator = actor_enumerator
        self.action_enumerator = action_enumerator
        if action_predictor is None:
            action_predictor = RandomActionPredictor()
        self.action_predictor = action_predictor
        self.forward_transitor = forward_transitor
        self.num_rollouts = num_rollouts
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

    def evaluate(self, state: State):
        '''
        Predicts the value of the state

        Args:
            state: current state

        Returns:
            value: value of the state
        '''
        values_estimates = np.zeros(self.num_rollouts)
        for i in range(self.num_rollouts):
            # copy state
            state = state.copy()
            # rollout
            while not state.is_done():

                # print('rollout state', state)
                # enumerate actors
                actors = self.actor_enumerator.enumerate(state)
                # print('rollout actors', actors)

                joint_action = dict()

                # for each actor get random action, add to joint action
                for actor in actors:
                    # print('rollout actor', actor)
                    # enumerate actions
                    actions = self.action_enumerator.enumerate(state, actor)
                    # print('rollout actions', actions)
                    # predict action probabilities
                    probs = self.action_predictor.predict(state, actions, actor)
                    # choose random action according to probs
                    action = self.random_state.choice(list(probs.keys()), p=list(probs.values()))
                    # add to joint action
                    joint_action[actor] = action

                # step
                state = self.forward_transitor.transition(state, joint_action)

            # get reward. TODO: only implemented for end states
            values_estimates[i] = state.get_reward()

        utility = np.mean(values_estimates)
        # print('rollout utility', utility)
        return utility
            


