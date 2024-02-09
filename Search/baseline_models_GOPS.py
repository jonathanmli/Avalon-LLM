import re
from typing import Dict, List
from Search.headers import *
from Search.prompts import *

# Parse helper funcs
def parse_bracketed_list(string: str) -> List[str]:
    pattern = r'\[([^\]]+)\]'

    matches = re.findall(pattern, string)

    items = [item.strip() for item in matches[0].split(',')] if matches else []

    return items

def parse_dict_with_any_key(text):
    pattern = r'\{.*?\}'

    matches = re.findall(pattern, text)

    return matches[-1]

def parse_int_value(string: str) -> int:
    pattern = r'\b\d+\b'

    integers = [int(num) for num in re.findall(pattern, string)]

    return integers[-1] if len(integers) > 0 else None # should be designed in the prompt that the last num is the value

def parse_prob_value(string: str) -> float:
    pattern = r'\b\d+\.\d+|\b\d+|\.\d+\b'

    floats = [float(num) for num in re.findall(pattern, string)]

    return floats[-1] if len(floats) > 0 else None


class GOPSState(State):
    '''
    GOPS state convention:

    ((1,2,5), (2,3), (4,1), 6, (0,1))

    (1,2,5 )are the prize cards shown to the player in that order
    (2,3) are the cards that the player has played in that order
    (4,1) are the cards that the opponent has played in that order
    6 is the total number of cards in each deck or hand
    (0,1) are the current players (0 for player, 1 for opponent)
    '''

    def __init__(self, actors, prize_cards, player_cards, opponent_cards, num_cards, done=False, reward=0.0):
        self.prize_cards = tuple(prize_cards)
        self.player_cards = tuple(player_cards)
        self.opponent_cards = tuple(opponent_cards)
        self.num_cards = int(num_cards)

        # should call super first, otherwise state_type will be overwritten
        id = tuple([self.prize_cards, self.player_cards, self.opponent_cards, self.num_cards])

        super().__init__(id, actors=actors, done=done, reward=reward)

        

    def copy(self):
        '''
        Returns a copy of the state
        '''
        return GOPSState(self.actors, self.prize_cards, self.player_cards, 
                         self.opponent_cards, self.num_cards, self.done, self.reward)
    
    def switch_copy(self):
        '''
        Returns a new state with the two players switched
        '''
        return GOPSState(self.actors, self.prize_cards, self.opponent_cards,
                         self.player_cards, self.num_cards, self.done, self.reward)

    
    def calculate_score(self):
        '''
        Calculates the score of the state for both players
        '''
        contested_points = 0
        player_score = 0
        opponent_score = 0
        for idx, single_score in enumerate(list(self.prize_cards)):
            contested_points += single_score
            if self.player_cards[idx] > self.opponent_cards[idx]:
                player_score += contested_points
                contested_points = 0
            elif self.player_cards[idx] < self.opponent_cards[idx]:
                opponent_score += contested_points
                contested_points = 0
        return (player_score, opponent_score)
        
class GOPSForwardTransitor(ForwardTransitor):

    def __init__(self):
        super().__init__()

    def transition(self, state: GOPSState, actions: dict):
        '''
        Transitions to the next state given the current state and action

        Args:
            state: current state
            actions: actions taken by the protagonist and the antagonist (ie cards played by the player and the opponent), or prize card played by the environment

        Returns:
            next_state: next state
        '''

        # we need to be careful to copy the state, otherwise the state will be changed
        prize_cards = state.prize_cards
        player_cards = state.player_cards
        opponent_cards = state.opponent_cards
        num_cards = state.num_cards
        actors = state.actors
        done = state.done
        reward = 0.0

        # assert that the keys of actions are the same as actors
        assert set(actions.keys()) == set(actors)

        if 0 in actors and 1 in actors: # simultaneous state
            # assert that len of actions is 2
            assert len(actions) == 2

            # assert that actions are not in the player_cards and between 1 and num_cards
            assert actions[0] not in player_cards and actions[0] in range(1, num_cards+1)

            # assert that actions are not in the opponent_cards and between 1 and num_cards
            assert actions[1] not in opponent_cards and actions[1] in range(1, num_cards+1)

            # append actions to player_cards and opponent_cards
            player_cards = list(player_cards)
            player_cards.append(actions[0])
            player_cards = tuple(player_cards)

            opponent_cards = list(opponent_cards)
            opponent_cards.append(actions[1])
            opponent_cards = tuple(opponent_cards)

            # change actors to {-1}
            actors = frozenset({-1})

            # check if the game is done
            if len(prize_cards) >= num_cards:
                actors = frozenset()
                done = True
                
                # calculate the score
                contested_points = 0
                player_score = 0
                opponent_score = 0
                for idx, single_score in enumerate(list(state.prize_cards)):
                    contested_points += single_score
                    if player_cards[idx] > opponent_cards[idx]:
                        player_score += contested_points
                        contested_points = 0
                    elif player_cards[idx] < opponent_cards[idx]:
                        opponent_score += contested_points
                        contested_points = 0
                
                reward = player_score - opponent_score
                        
        elif -1 in actors: # random state
            # assert that len of actions is 1
            assert len(actions) == 1

            # print('actions', actions)
            # print('prize_cards', prize_cards)

            # assert that actions are not in the prize_cards and between 1 and num_cards
            assert actions[-1] not in prize_cards and actions[-1] in range(1, num_cards+1)

            # append actions to prize_cards
            prize_cards = list(prize_cards)
            prize_cards.append(actions[-1])
            prize_cards = tuple(prize_cards)

            # change actors to {0, 1}
            actors = frozenset({0, 1})

        else:
            raise ValueError('Invalid actors'+str(actors))
        

        out_state = GOPSState(actors, prize_cards, player_cards, opponent_cards, num_cards, done, reward)
        return out_state

class GOPSActionEnumerator(ActionEnumerator):

    def __init__(self):
        super().__init__()

    def enumerate(self, state: GOPSState, actor) -> set:
        '''
        Enumerates the possible actions that the player can take given the current state

        Args:
            state: current state
            actor: actor to enumerate actions for

        Returns:
            actions: set of actions
        '''
        if actor == 0:
            player_cards = state.player_cards
            num_cards = state.num_cards
            starting_deck = list(range(1, num_cards+1))
            actions = set(starting_deck) - set(player_cards)
            return actions
        elif actor == 1:
            opponent_cards = state.opponent_cards
            num_cards = state.num_cards
            starting_deck = list(range(1, num_cards+1))
            actions = set(starting_deck) - set(opponent_cards)
            return actions
        elif actor == -1:
            prize_cards = state.prize_cards
            num_cards = state.num_cards
            starting_deck = list(range(1, num_cards+1))
            actions = set(starting_deck) - set(prize_cards)
            return actions
        else:
            raise ValueError('Invalid actor'+str(actor))
    
class GOPSRandomActionPredictor(ActionPredictor):

    def __init__(self):
        super().__init__()

    def predict(self, state: GOPSState, actions, actor=-1):
        '''
        Predicts the probabilities over actions given the current state

        Args:
            state: current state
            actions: list of actions

        Returns:
            probs: dictionary of probabilities over actions
        '''
        # assert that actor is -1
        assert actor == -1

        probs = dict()
        for action in actions:
            probs[action] = 1.0/len(actions)
        return probs
    
class GOPSActorEnumerator(ActorEnumerator):

    def __init__(self):
        super().__init__()

    def enumerate(self, state: GOPSState) -> set:
        '''
        Enumerates the actors that may take actions at the state

        Args:
            state: current state

        Returns:
            actors: set of actors that may take actions at the state
        '''
        return state.actors
    
# TODO: refactor this
# class GPT35OpponentActionPredictor(OpponentActionPredictor):
#     '''
#     Opponent action predictor for GPT-3.5
#     '''

#     def __init__(self, model):
#         '''
#         Args:
#             model: GPT-3.5 model
#         '''
#         self.model = model

#     def predict(self, state: GOPSState, actions, player=0, prob=True) -> Dict:
#         '''
#         Predicts the advantage of each opponent action given the current state and action

#         Args:
#             state: current state
#             actions: set or list of actions

#         Returns:
#             advantage: list of relative advantages of each opponent action (probs for current implementation)
#         '''
#         player_cards = state.player_cards
#         opponent_cards = state.opponent_cards
#         prize_cards = state.prize_cards

#         player_hand = [i for i in range(1, state.num_cards+1)]
#         opponent_hand = [i for i in range(1, state.num_cards+1)]
#         score_cards = [i for i in range(1, state.num_cards+1)]

#         player_hand = list(set(player_hand) - set(player_cards))
#         opponent_hand = list(set(opponent_hand) - set(opponent_cards))
#         score_cards = list(set(prize_cards) - set(score_cards))

#         # print(actions)

#         # verbalized_opaction_prompt = VERBALIZED_OPACTION_PREDICTOR.format(
#         #     played_cards=prize_cards,
#         #     score_cards=player_cards,
#         #     your_cards=opponent_cards,
#         #     your_hand=player_hand,
#         #     opponent_cards=opponent_cards,
#         #     opponent_hand=opponent_hand,
#         #     # your_score=player_score,
#         #     # opponent_score=opponent_score,
#         #     opponent_actions=actions
#         # )
#         # TODO: fix OPPONENT_ACTION_PREDICTOR_PROMPT to be better

#         # Uncomment the following to use the model

#         # Call the model
#         # output = self.model.single_action(verbalized_opaction_prompt)

#         # # Parse the output
#         # try:
#         #     advantages = eval(parse_dict_with_any_key(output))
#         #     advantages = {int(key): value for key, value in advantages.items()}
#         #     assert len(advantages) == len(actions)
#         # except:
#         #     advantages = {action: 1.0/len(actions) for action in actions}

#         import random
#         advantages = {action: (1.0 * random.randint(0, len(actions)))/len(actions) for action in actions}
#         print(advantages)

#         # print(advantages)

#         return advantages

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

    def evaluate(self, state: GOPSState) -> Dict:
        '''
        Predicts the value of the state

        Args:
            state: current state

        Returns:
            value: value of the state
        '''
        # # Prepare input
        # prob_prompt = "Current State: {state}\n".format(state=state.notes)
        # prob_prompt += VALUE_PREDICTOR_PROMPTS[0]
        # value_prompt = "Current State: {state}\n".format(state=state.notes)
        # value_prompt += VALUE_PREDICTOR_PROMPTS[1]

        player_cards = state.player_cards
        opponent_cards = state.opponent_cards
        prize_cards = state.prize_cards

        # Calculate the score for the state
        contested_score = 0
        player_score = 0
        opponent_score = 0
        for idx, single_score in enumerate(list(state.prize_cards)):
            contested_score += single_score
            if player_cards[idx] > opponent_cards[idx]:
                player_score += contested_score
                contested_score = 0
            elif player_cards[idx] < opponent_cards[idx]:
                opponent_score += contested_score
                contested_score = 0
            elif player_cards[idx] == opponent_cards[idx]:
                contested_score += single_score

        player_hand = [i for i in range(1, state.num_cards+1)]
        opponent_hand = [i for i in range(1, state.num_cards+1)]
        score_cards = [i for i in range(1, state.num_cards+1)]

        player_hand = list(set(player_hand) - set(player_cards))
        opponent_hand = list(set(opponent_hand) - set(opponent_cards))
        score_cards = list(set(prize_cards) - set(score_cards))

        # verbalized_value_prompt = VERBALIZED_VALUE_PREDICOTR.format(
        #     played_cards=prize_cards,
        #     score_cards=player_cards,
        #     your_cards=opponent_cards,
        #     your_hand=player_hand,
        #     opponent_cards=opponent_cards,
        #     opponent_hand=opponent_hand,
        #     your_score=player_score,
        #     opponent_score=opponent_score
        # )

        # # Uncomment the following to use the model

        # # Call the model
        # prob_output = self.model.single_action(prob_prompt)
        # value_output = self.model.single_action(verbalized_value_prompt)

        # # Parse the output
        # # prob_value = parse_prob_value(prob_output)
        # # value = parse_int_value(value_output)

        # value = value_output


        # New Prompt Framework
        # 1. Representation Prompt
        current_state = STATE_PROMPT.format(
            played_cards=prize_cards,
            score_cards=player_cards,
            your_cards=opponent_cards,
            your_hand=player_hand,
            opponent_cards=opponent_cards,
            opponent_hand=opponent_hand,
            your_score=player_score,
            opponent_score=opponent_score
        )

        current_situation = self.model.single_action(f"{current_state}\n\n{REPRESENTATION_PROMPTS[0]}")
        # print(current_situation)

        # 2.a. Points earned so far Prompt
        POINTS_EARNED_SO_FAR_PROMPT = """Given the current situation, how many points have you won so far? Write down your thoughts and output the number of points."""
        points_earned_so_far = self.model.single_action(f"Current Situation: {current_situation}\n\n{POINTS_EARNED_SO_FAR_PROMPT}")

        # 2.b. Expected points to win in future Prompt
        EXPECTED_POINTS_TO_WIN_PROMPT = """Given the current situation, how many more points do you expect to win in the future? Write down your thoughts and output the number of points."""
        expected_points_to_win = self.model.single_action(f"Current Situation: {current_situation}\n\n{EXPECTED_POINTS_TO_WIN_PROMPT}")

        # 3. Sum total points to win Prompt
        SUM_TOTAL_POINTS_TO_WIN_PROMPT = """Given the current situation, how many points do you expect to get at the end of the game? Write down your thoughts and output the number of points."""
        sum_prompt = f"Points Earned So Far: {points_earned_so_far}\n\nExpected Points to Win: {expected_points_to_win}\n\n{SUM_TOTAL_POINTS_TO_WIN_PROMPT}"
        sum_output = self.model.single_action(sum_prompt)

        # 4. Parser (str -> int)
        value_output = self.model.single_action(sum_output)

        # Parse the output
        # prob_value = parse_prob_value(prob_output)
        value = parse_int_value(value_output)

        # TODO: optimize this maybe
        if not isinstance(value, int):
            value = 5

        # print(f"State: {state} Value: {value}")

        return value
    

class LLMFunctionalValueHeuristic(ValueHeuristic):
    '''
    Functional value heuristic for LLMs
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model

        # feed the model both the rules of game and the heuristics function prompt
        prompt1 = GOPS_RULES + '\n' + HEURISTICS_FUNCTION_PROMPTS[0]
        abstract_function = self.model.single_action(prompt1)

        # now feed the both previous prompt and response and the GOPS_VALUE_FUNCTION_PROMPT
        prompt2 = prompt1 + '\n' + abstract_function + '\n' + GOPS_VALUE_FUNCTION_PROMPT
        function = self.model.single_action(prompt2)

        # exec the function
        exec(function)

        self._evaluate = evaluate_state


    def evaluate(self, state: GOPSState) -> Dict:
        '''
        Predicts the value of the state

        Args:
            state: current state

        Returns:
            value: value of the state
        '''
        # Prepare input
        player_cards = state.player_cards
        opponent_cards = state.opponent_cards
        prize_cards = state.prize_cards
        (player_score, opponent_score) = state.calculate_score()
        is_player_turn = 0 in state.actors

        # use the function to calculate the value
        value = self._evaluate((prize_cards, player_cards, opponent_cards, is_player_turn, player_score, opponent_score))
        return value