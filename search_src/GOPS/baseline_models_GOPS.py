import os
import re
from typing import Dict, List
from search_search_src.searchlight.headers import *
# from search_search_src.search_src.prompts import *
import search_search_src.searchlight.utils as utils
from search_search_src.searchlightimprove.prompts.improvement_prompts import *
# import logging

# FIXME: logging config should be defined in main file (run_*.py)
# if not os.path.exists('./output'):
#     os.makedirs("./output")
# i = len([f for f in os.listdir("./output/") if os.path.isfile(os.path.join("./output", f))])
# logging.basicConfig(
#     filename=f"./output/funciton_{i}.log",
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )

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


class GOPSState2(State):
    '''
    GOPS state convention:

    ((1,2,5), (2,3), (4,1), 6, (0,1))

    (1,2,5 )are the prize cards shown to the player in that order
    (2,3) are the cards that the player has played in that order
    (4,1) are the cards that the opponent has played in that order
    6 is the total number of cards in each deck or hand
    (0,1) are the current players (0 for player, 1 for opponent)
    '''

    def __init__(self, actors, prize_cards, player_cards, opponent_cards, num_cards, done=False):
        self.prize_cards = tuple(prize_cards)
        self.player_cards = tuple(player_cards)
        self.opponent_cards = tuple(opponent_cards)
        self.num_cards = int(num_cards)
        self.actors = actors
        self.done = done

        # should call super first, otherwise state_type will be overwritten
        id = tuple([self.prize_cards, self.player_cards, self.opponent_cards, self.num_cards])

        super().__init__(id, notes=actors)

    def is_done(self):
        '''
        Returns whether the game is done
        '''
        return self.done

    def copy(self):
        '''
        Returns a copy of the state
        '''
        return GOPSState2(self.actors, self.prize_cards, self.player_cards, 
                         self.opponent_cards, self.num_cards, self.done)
    
    def switch_copy(self):
        '''
        Returns a new state with the two players switched
        '''
        return GOPSState2(self.actors, self.prize_cards, self.opponent_cards,
                         self.player_cards, self.num_cards, self.done)
    
    def get_reward(self):
        '''
        Returns the reward of the state
        '''
        player_score, opponent_score = self.calculate_score()
        return player_score - opponent_score
    
    def get_score_deck(self):
        '''
        Returns the score deck
        '''
        return set(range(1, self.num_cards+1)) - set(self.prize_cards)

    def get_player_hand(self):
        '''
        Returns the player's hand
        '''
        return set(range(1, self.num_cards+1)) - set(self.player_cards)
    
    def get_opponent_hand(self):
        '''
        Returns the opponent's hand
        '''
        return set(range(1, self.num_cards+1)) - set(self.opponent_cards)
    
    def calculate_score(self):
        '''
        Calculates the score of the state for both players
        '''
        contested_points = 0
        player_score = 0
        opponent_score = 0
        for idx, single_score in enumerate(list(self.prize_cards)):
            contested_points += single_score
            # if idx >= len(self.player_cards) or idx >= len(self.opponent_cards), break
            if idx >= len(self.player_cards) or idx >= len(self.opponent_cards):
                break
            if self.player_cards[idx] > self.opponent_cards[idx]:
                player_score += contested_points
                contested_points = 0
            elif self.player_cards[idx] < self.opponent_cards[idx]:
                opponent_score += contested_points
                contested_points = 0
        return (player_score, opponent_score)
    
    def get_scores(self):
        '''
        Returns the scores of the state for both players
        '''
        return self.calculate_score()
    
GOPS_START_STATE_6 = GOPSState2({-1}, tuple(), tuple(), tuple(), 6, False)
        
class GOPSForwardTransitor2(ForwardTransitor2):

    def __init__(self):
        super().__init__()

    def _transition(self, state: GOPSState2, actions: dict) -> tuple[Optional[State], dict[Any, float], dict]:
        # we need to be careful to copy the state, otherwise the state will be changed
        prize_cards = state.prize_cards
        player_cards = state.player_cards
        opponent_cards = state.opponent_cards
        num_cards = state.num_cards
        actors = state.actors
        done = state.done
        reward = {0: 0.0, 1: 0.0}

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

            # check if the game is almost done
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
                
                reward[0] = player_score - opponent_score
                reward[1] = opponent_score - player_score
                        
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
        elif not actors: # terminal state
            return None, reward, dict()
        else:
            print(state)
            raise ValueError('Invalid actors'+str(actors))
        

        out_state = GOPSState2(actors, prize_cards, player_cards, opponent_cards, num_cards, done)
        return out_state, reward, dict()

class GOPSActionEnumerator(ActionEnumerator):

    def __init__(self):
        super().__init__()

    def _enumerate(self, state: GOPSState2, actor) -> set:
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
    
class GOPSRandomActionPredictor(PolicyPredictor):

    def __init__(self):
        super().__init__()

    def _predict(self, state: GOPSState2, actions, actor=-1):
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

    def _enumerate(self, state: GOPSState2) -> set:
        '''
        Enumerates the actors that may take actions at the state

        Args:
            state: current state

        Returns:
            actors: set of actors that may take actions at the state
        '''
        # print('state', state)
        # print('actors', state.actors)
        return state.actors
    
# class ZeroSumValueHeurtisticWrapper(ValueHeuristic2):
#     '''
#     Wrapper for zero sum value heuristic
#     '''

#     def __init__(self, value_heuristic: ValueHeuristic2):
#         self.value_heuristic = value_heuristic

#     def evaluate(self, state: GOPSState2):
#         '''
#         Evaluates the state

#         Args:
#             state: current state

#         Returns:
#             value: value of the state
#         '''
#         # print(state)
#         value, notes = self.value_heuristic.evaluate(state, actor=0)
#         return {0: value, 1: -value}, notes
    
class GOPSInitialInferencer2(InitialInferencer2):

    def __init__(self, transitor: ForwardTransitor2, action_enumerator: ActionEnumerator, 
                 action_predictor: PolicyPredictor, actor_enumerator: ActorEnumerator,
                 value_heuristic: ValueHeuristic2):
        super().__init__()
        self.transitor = transitor
        self.action_enumerator = action_enumerator
        self.action_predictor = action_predictor
        self.actor_enumerator = actor_enumerator
        self.value_heuristic = value_heuristic

    def _predict(self, state: GOPSState2) -> tuple[dict, dict, dict[tuple[tuple[Any, Any],...],Any], dict[tuple[tuple[Any, Any],...],Any], dict[State,dict]]:
        # predict actors using actor_enumerator
        actors = self.actor_enumerator.enumerate(state)
        # predict actions using action_enumerator for each actor
        actor_to_actions = {actor: self.action_enumerator.enumerate(state, actor) for actor in actors}
        # predict probs using action_predictor for each actor
        actor_to_action_to_probs = {actor: self.action_predictor.predict(state, actor_to_actions[actor], actor) for actor in actors}
        # get joint actions from actor_to_actions. joint actions should be tuples of tuples (actor, action), i.e. joint_action1 = ((actor1, action1), (actor2, action2))
        # joint actions should contain cartesian product of actions for each actor
        joint_actions = utils.dict_to_set_of_cartesian_products(actor_to_actions)

        notes = dict()

        if (actors is None) or (not actors):
            joint_action_to_next_state = dict()
            next_state_to_value = dict()
            joint_action_to_rewards = dict()
            next_state_to_notes = dict()
        else:
            # get transitioned states from transitor for each joint action
            joint_action_to_next_state_rewards_notes = {joint_action: self.transitor.transition(state, {actor: action for actor, action in joint_action}) for joint_action in joint_actions}
            joint_action_to_next_state = {joint_action: joint_action_to_next_state_rewards_notes[joint_action][0] for joint_action in joint_actions}
            joint_action_to_rewards = {joint_action: joint_action_to_next_state_rewards_notes[joint_action][1] for joint_action in joint_actions}

            # get value of each next state using value_heuristic
            next_state_to_value_notes = {next_state: self.value_heuristic.evaluate(next_state) for next_state in joint_action_to_next_state.values()}
            next_state_to_value = {next_state: next_state_to_value_notes[next_state][0] for next_state in joint_action_to_next_state.values()}
            next_state_to_notes = {next_state: next_state_to_value_notes[next_state][1] for next_state in joint_action_to_next_state.values()}
        
        notes['next_state_to_heuristic_notes'] = next_state_to_notes
        return actor_to_action_to_probs, next_state_to_value, joint_action_to_rewards, joint_action_to_next_state, notes
    
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

# TODO: refactor this
# class GPT35ValueHeuristic(ValueHeuristic):
#     '''
#     Value heuristic for GPT-3.5
#     '''

#     def __init__(self, model):
#         '''
#         Args:
#             model: GPT-3.5 model
#         '''
#         self.model = model

#     def evaluate(self, state: GOPSState) -> Dict:
#         '''
#         Predicts the value of the state

#         Args:
#             state: current state

#         Returns:
#             value: value of the state
#         '''
#         # # Prepare input
#         # prob_prompt = "Current State: {state}\n".format(state=state.notes)
#         # prob_prompt += VALUE_PREDICTOR_PROMPTS[0]
#         # value_prompt = "Current State: {state}\n".format(state=state.notes)
#         # value_prompt += VALUE_PREDICTOR_PROMPTS[1]

#         player_cards = state.player_cards
#         opponent_cards = state.opponent_cards
#         prize_cards = state.prize_cards

#         # Calculate the score for the state
#         contested_score = 0
#         player_score = 0
#         opponent_score = 0
#         for idx, single_score in enumerate(list(state.prize_cards)):
#             contested_score += single_score
#             if player_cards[idx] > opponent_cards[idx]:
#                 player_score += contested_score
#                 contested_score = 0
#             elif player_cards[idx] < opponent_cards[idx]:
#                 opponent_score += contested_score
#                 contested_score = 0
#             elif player_cards[idx] == opponent_cards[idx]:
#                 contested_score += single_score

#         player_hand = [i for i in range(1, state.num_cards+1)]
#         opponent_hand = [i for i in range(1, state.num_cards+1)]
#         score_cards = [i for i in range(1, state.num_cards+1)]

#         player_hand = list(set(player_hand) - set(player_cards))
#         opponent_hand = list(set(opponent_hand) - set(opponent_cards))
#         score_cards = list(set(prize_cards) - set(score_cards))

#         # verbalized_value_prompt = VERBALIZED_VALUE_PREDICOTR.format(
#         #     played_cards=prize_cards,
#         #     score_cards=player_cards,
#         #     your_cards=opponent_cards,
#         #     your_hand=player_hand,
#         #     opponent_cards=opponent_cards,
#         #     opponent_hand=opponent_hand,
#         #     your_score=player_score,
#         #     opponent_score=opponent_score
#         # )

#         # # Uncomment the following to use the model

#         # # Call the model
#         # prob_output = self.model.single_action(prob_prompt)
#         # value_output = self.model.single_action(verbalized_value_prompt)

#         # # Parse the output
#         # # prob_value = parse_prob_value(prob_output)
#         # # value = parse_int_value(value_output)

#         # value = value_output


#         # New Prompt Framework
#         # 1. Representation Prompt
#         current_state = STATE_PROMPT.format(
#             played_cards=prize_cards,
#             score_cards=player_cards,
#             your_cards=opponent_cards,
#             your_hand=player_hand,
#             opponent_cards=opponent_cards,
#             opponent_hand=opponent_hand,
#             your_score=player_score,
#             opponent_score=opponent_score
#         )

#         current_situation = self.model.single_action(f"{current_state}\n\n{REPRESENTATION_PROMPTS[0]}")
#         # print(current_situation)

#         # 2.a. Points earned so far Prompt
#         POINTS_EARNED_SO_FAR_PROMPT = """Given the current situation, how many points have you won so far? Write down your thoughts and output the number of points."""
#         points_earned_so_far = self.model.single_action(f"Current Situation: {current_situation}\n\n{POINTS_EARNED_SO_FAR_PROMPT}")

#         # 2.b. Expected points to win in future Prompt
#         EXPECTED_POINTS_TO_WIN_PROMPT = """Given the current situation, how many more points do you expect to win in the future? Write down your thoughts and output the number of points."""
#         expected_points_to_win = self.model.single_action(f"Current Situation: {current_situation}\n\n{EXPECTED_POINTS_TO_WIN_PROMPT}")

#         # 3. Sum total points to win Prompt
#         SUM_TOTAL_POINTS_TO_WIN_PROMPT = """Given the current situation, how many points do you expect to get at the end of the game? Write down your thoughts and output the number of points."""
#         sum_prompt = f"Points Earned So Far: {points_earned_so_far}\n\nExpected Points to Win: {expected_points_to_win}\n\n{SUM_TOTAL_POINTS_TO_WIN_PROMPT}"
#         sum_output = self.model.single_action(sum_prompt)

#         # 4. Parser (str -> int)
#         value_output = self.model.single_action(sum_output)

#         # Parse the output
#         # prob_value = parse_prob_value(prob_output)
#         value = parse_int_value(value_output)

#         # TODO: optimize this maybe
#         if not isinstance(value, int):
#             value = 5

#         # print(f"State: {state} Value: {value}")

#         return value
    

class LLMFunctionalValueHeuristic(ValueHeuristic2):
    '''
    Functional value heuristic for LLMs
    '''

    EVAL_TEST_STATES = [
        ((1, 3, 2, 6, 5), (2, 3, 4, 1, 6), (1, 3, 4, 6, 5), False, 6, 11, {4,}, {5,}, {2,}),
        ((1, 3, 2), (2, 3, 4), (1, 3, 2), False, 6, 0, {4, 5, 6}, {1, 5, 6}, {2, 4, 6}),
        ((1, 3, 2, 4), (2, 3, 1, 4), (1, 3, 2, 6), False, 1, 9, {5, 6}, {5, 6}, {4, 5}),
        ((1, 3), (2, 3), (1, 2), False, 4, 0, {2, 4, 5, 6}, {1, 4, 5, 6}, {3, 4, 5, 6}),
        ((1, 3, 6), (2, 1, 4), (1, 3, 5), False, 1, 9, {2, 4, 5}, {3, 5, 6}, {2, 4, 6}),
        ((1,), (1,), (6,), False, 0, 1, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {1, 2, 3, 4, 5}),
        ((1, 4), (1, 3), (2, 5), False, 0, 5, {2, 3, 5, 6}, {2, 4, 5, 6}, {1, 3, 4, 6}),
        ((4, 3, 1, 2, 6), (2, 3, 4, 1, 6), (2, 3, 4, 5, 1), False, 6, 10, {5,}, {5,}, {6,}),
        ((4, 3), (2, 1), (2, 4), False, 0, 7, {1, 2, 5, 6}, {3, 4, 5, 6}, {1, 3, 5, 6}),
        ((4,), (2,), (5,), False, 0, 4, {1, 2, 3, 5, 6}, {1, 3, 4, 5, 6}, {1, 2, 3, 4, 6}),
        ((4, 3, 1, 2, 5), (2, 3, 4, 1, 6), (2, 3, 4, 6, 5), False, 5, 10, {6,}, {5,}, {1,}),
        ((4,), (3,), (2,), False, 4, 0, {1, 2, 3, 5, 6}, {1, 2, 4, 5, 6}, {1, 3, 4, 5, 6}),
        ((4, 2, 6, 5, 1, 3), (6, 5, 4, 3, 2), (6, 5, 4, 3, 2), True, 0, 0, set(), {1}, {1}),
        ((3, 1, 2), (3, 2, 1), (3, 2, 1), False, 0, 0, set(), set(), set()), # NOTE: this is an end state
        ((1, 2), (1,), (1,), True, 0, 0, {3}, {2, 3}, {2, 3}),
    ]



    def __init__(self, model=None, func=None):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model

        if func is None:
            # keep generating functions until we pass the test
            passed = False
            while not passed:
                # feed the model both the rules of game and the heuristics function prompt
                prompt1 = gen_seed_thought_prompt()
                abstract_function = self.model.single_action(prompt1)

                # print the abstract function for debugging
                # print('abstract function: \n', abstract_function)

                # now feed the both previous prompt and response and the GOPS_VALUE_FUNCTION_PROMPT
                prompt2 = gen_seed_function_prompt(abstract_function)
                function_str = self.model.single_action(prompt2)

                # print the function for debugging
                # print('function: \n', function_str)

                passed = self.test_evaluate(function_str)
        else:
            function_str = func
            passed = self.test_evaluate(function_str)

        self.passed = passed
        # if passed:
        #     print('passed function', function_str)
        # else:
        #     print('failed function', function_str)

        # # parse out ```python ... ``` from the response
        # pattern = r'```python(.*?)```'
        # matches = re.findall(pattern, function, re.DOTALL)
        # print('matches', matches)
        # function = matches[-1].strip()
    
    def test_evaluate(self, function_str, safe_mode = True) -> bool:
        '''
        Test the evaluate function
        '''
        # TODO: this should be a static method
        def parse_function(function_str):
            # Parse the function definition
            pattern = re.compile(r'(?ms)^def\s+evaluate_state\s*\(.*?\)\s*.*?:\s*.*?(?=^\w|\Z)', re.MULTILINE)
            match = re.search(pattern, function_str)
            if match:
                # print(f"Parsed Function:\n{match.group()}")
                # logging.info(match.group()) #TODO: fix logging to match the new logging framework
                parsed_func = match.group()
                function_lines = parsed_func.split('\n')
                last_line_id = -2
                for line_id in range(len(function_lines)-1, -1, -1):
                    if function_lines[line_id].lstrip().startswith("return"):
                        last_line_id = line_id
                        break
                return '\n'.join(function_lines[:last_line_id+1])
            else:
                # print("Function not found!!!!!!!!")
                if not safe_mode:
                    raise ValueError("Parsing error: Function not found")
                return None
            
        if safe_mode:
            # Test the evaluate function
            try:
                parsed_function = parse_function(function_str)
                if parsed_function is None:
                    return False
                # Execute the function definition within the local scope of __init__
                exec(parsed_function, globals(), locals())
                
                # Attach the dynamically defined function to the instance
                self._llm_evaluate = locals()['evaluate_state']

                # print('successfully defined function')

                for state in self.EVAL_TEST_STATES:
                    (player_value, opponent_value), notes = self._llm_evaluate(state)
                    # assert that both values are numbers
                    assert isinstance(player_value, (int, float))
                    assert isinstance(opponent_value, (int, float))
                    # assert that notes is a dictionary
                    assert isinstance(notes, dict)

                # print('successfully passed test')
            except Exception as e:  # Capture the exception as 'e'
                print(f"An exception occurred: {e}")  # Print the exception for debugging
                return False
            return True
        else:
            parsed_function = parse_function(function_str)
            if parsed_function is None:
                return False
            # Execute the function definition within the local scope of __init__
            exec(parsed_function, globals(), locals())
            
            # Attach the dynamically defined function to the instance
            self._llm_evaluate = locals()['evaluate_state']

            # print('successfully defined function')

            for state in self.EVAL_TEST_STATES:
                (player_value, opponent_value), notes = self._llm_evaluate(state)
                # assert that both values are numbers
                assert isinstance(player_value, (int, float))
                assert isinstance(opponent_value, (int, float))
                # assert that notes is a dictionary
                assert isinstance(notes, dict)

            return True

    @staticmethod
    def test_evaluate_static(function_str, safe_mode = False) -> bool:
        '''
        Test the evaluate function
        '''
        def parse_function(function_str):
            # Parse the function definition
            pattern = re.compile(r'(?ms)^def\s+evaluate_state\s*\(.*?\)\s*.*?:\s*.*?(?=^\w|\Z)', re.MULTILINE)
            match = re.search(pattern, function_str)
            if match:
                # print(f"Parsed Function:\n{match.group()}")
                # logging.info(match.group()) #TODO: fix logging to match the new logging framework
                parsed_func = match.group()
                function_lines = parsed_func.split('\n')
                last_line_id = -2
                for line_id in range(len(function_lines)-1, -1, -1):
                    if function_lines[line_id].lstrip().startswith("return"):
                        last_line_id = line_id
                        break
                return '\n'.join(function_lines[:last_line_id+1])
            else:
                # print("Function not found!!!!!!!!")
                if not safe_mode:
                    raise ValueError("Parsing error: Function not found")
                return None
            
        if safe_mode:
            # Test the evaluate function
            try:
                parsed_function = parse_function(function_str)
                if parsed_function is None:
                    return False
                # Execute the function definition within the local scope of __init__
                exec(parsed_function, globals(), locals())
                
                # Attach the dynamically defined function to the instance
                llm_evaluate = locals()['evaluate_state']

                # print('successfully defined function')

                for state in LLMFunctionalValueHeuristic.EVAL_TEST_STATES:
                    (player_value, opponent_value), notes = llm_evaluate(state)
                    # assert that both values are numbers
                    assert isinstance(player_value, (int, float))
                    assert isinstance(opponent_value, (int, float))
                    # assert that notes is a dictionary
                    assert isinstance(notes, dict)

                # print('successfully passed test')
            except Exception as e:  # Capture the exception as 'e'
                print(f"An exception occurred: {e}")  # Print the exception for debugging
                return False
            return True
        else:
            parsed_function = parse_function(function_str)
            if parsed_function is None:
                return False
            # Execute the function definition within the local scope of __init__
            exec(parsed_function, globals(), locals())
            
            # Attach the dynamically defined function to the instance
            llm_evaluate = locals()['evaluate_state']

            # print('successfully defined function')

            for state in LLMFunctionalValueHeuristic.EVAL_TEST_STATES:
                (player_value, opponent_value), notes = llm_evaluate(state)
                # assert that both values are numbers
                assert isinstance(player_value, (int, float))
                assert isinstance(opponent_value, (int, float))
                # assert that notes is a dictionary
                assert isinstance(notes, dict)

            return True
        
    @staticmethod
    def parse_llm_function(function_str: str, safe_mode = False) -> str:
        # Parse the function definition
        pattern = re.compile(r'(?ms)^def\s+evaluate_state\s*\(.*?\)\s*.*?:\s*.*?(?=^\w|\Z)', re.MULTILINE)
        match = re.search(pattern, function_str)
        if match:
            # print(f"Parsed Function:\n{match.group()}")
            # logging.info(match.group()) #TODO: fix logging to match the new logging framework
            parsed_func = match.group()
            function_lines = parsed_func.split('\n')
            last_line_id = -2
            for line_id in range(len(function_lines)-1, -1, -1):
                if function_lines[line_id].lstrip().startswith("return"):
                    last_line_id = line_id
                    break
            return '\n'.join(function_lines[:last_line_id+1])
        else:
            # print("Function not found!!!!!!!!")
            if not safe_mode:
                raise ValueError("Parsing error: Function not found")
            return None
        

    def _evaluate(self, state: GOPSState2) -> tuple[dict, dict]:
        # Prepare input
        player_cards = state.player_cards
        opponent_cards = state.opponent_cards
        prize_cards = state.prize_cards
        (player_score, opponent_score) = state.calculate_score()
        is_player_turn = 0 in state.actors
        remain_prize_cards = state.get_score_deck()
        player_hand = state.get_player_hand()
        opponent_hand = state.get_opponent_hand()

        # make sure the inputs are of the correct type
        assert isinstance(player_cards, tuple)
        assert isinstance(opponent_cards, tuple)
        assert isinstance(prize_cards, tuple)
        # assert isinstance(player_score, int)
        # assert isinstance(opponent_score, int)
        assert isinstance(is_player_turn, bool)
        assert isinstance(remain_prize_cards, set)
        assert isinstance(player_hand, set)
        assert isinstance(opponent_hand, set)


        # use the function to calculate the value
        try:
            # print input for debugging
            # print('input')
            # print(tuple([prize_cards, player_cards, opponent_cards, is_player_turn, player_score, opponent_score, num_cards, remain_prize_cards]))
            
            state_tup = (prize_cards, player_cards, opponent_cards, is_player_turn, player_score, opponent_score, remain_prize_cards, player_hand, opponent_hand)
            # print(state_tup)
            (player_value, opponent_value), notes = self._llm_evaluate(state_tup)
        # raise an error if the function is not defined properly
        except Exception as e:
            logging.warning(f"state tuple: {str(state_tup)}")
            # print('state tuple', state_tup)
            # NOTE: add any printed tuples to EVAL_TEST_STATES for future testing
            raise ValueError(f"Function not defined properly: {e}")
        
        return {0:player_value - opponent_value, 1:opponent_value - player_value}, notes