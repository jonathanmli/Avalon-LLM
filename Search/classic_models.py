from Search.headers import *
import numpy as np

class RandomRolloutValueHeuristic(ValueHeuristic):

    def __init__(self, action_enumerator: ActionEnumerator, opponent_action_enumerator: OpponentActionEnumerator,
                 forward_transitor: ForwardTransitor, rs_enumerator:RandomStateEnumerator, num_rollouts=100):
        '''
        Args:
            action_enumerator: action enumerator
            opponent_action_enumerator: opponent action enumerator
            num_rollouts: number of rollouts to perform
        '''
        super().__init__()
        self.action_enumerator = action_enumerator
        self.opponent_action_enumerator = opponent_action_enumerator
        self.forward_transitor = forward_transitor
        self.rs_enumerator = rs_enumerator
        self.num_rollouts = num_rollouts

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
                if state.state_type == 'simultaneous': # only works or 1 oppoenent atm
                    # get actions
                    actions = self.action_enumerator.enumerate(state)
                    opponent_actions = self.opponent_action_enumerator.enumerate(state)
                    # choose random action
                    action = np.random.choice(list(actions))
                    opponent_action = np.random.choice(list(opponent_actions))
                    joint_action = (action, opponent_action)
                    # step
                    state = self.forward_transitor.transition(state, joint_action)
                elif state.state_type == 'stochastic':
                    # get actions
                    actions = self.rs_enumerator.enumerate(state)
                    # print('state', state)
                    # print('actions', actions)
                    # print('state type', state_copy.state_type)
                    # choose random action
                    action = np.random.choice(actions)
                    # step
                    state = self.forward_transitor.transition(state, action)
                else:
                    raise NotImplementedError
            # get reward. TODO: only implemented for end states
            values_estimates[i] = state.get_reward()

        utility = np.mean(values_estimates)
        # print('rollout utility', utility)
        return utility
            


