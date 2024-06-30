from .improvement_prompts import *
from search_src.searchlight.utils import AbstractLogged
import re


class PromptGenerator(AbstractLogged):
    '''
    Class that contains all the generation methods for all LLM prompts for self improvement

    With all the prompts in one place, it is easier to make changes to the prompts and to keep track of the prompts and prompt tuning
    '''
    def __init__(self, environment_rules, function_signature, sys_prompt = SYS_PROMPT, seed_heuristic_thought_prompt = 0):
        self.game_rules = environment_rules
        self.function_signature = function_signature
        self.sys_prompt = sys_prompt
        self.seed_heuristic_thought_prompt = seed_heuristic_thought_prompt
        super().__init__()

    def gen_seed_thought_prompt(self):
        # combine SYS_PROMPT, GOPS_RULES, HEURISTICS_SEED_PROMPT, and GOPS_FUNCTION_SIGNATURE
        return self.sys_prompt + self.game_rules + HEURISTICS_SEED_THOUGHT_PROMPTS[self.seed_heuristic_thought_prompt]
    
    def gen_critic_score_prompt(self, func: str):
        return self.sys_prompt + self.game_rules + "\n Previously you generated the following function: \n" + func + """\n \n On a scale of 1 to 10, how well do you think this function will perform in evaluating the value of a state in the game? Please provide a brief explanation for your score. 
        
        You should use the following format for your score and explanation:
        
        Explanation: <your explanation here>
        
        Score: <float, your score here>"""
    
    @staticmethod
    def parse_critic_score(string: str) -> tuple[str, float]:
        # Regular expression to find patterns of "Explanation: <explanation here>"
        explanation_pattern = re.compile(r"Explanation: (.+)")
        
        # Regular expression to find patterns of "Score: <score here>"
        score_pattern = re.compile(r"Score: (\d+\.\d+)")
        
        # Find all matches of the pattern in the input string
        explanation = explanation_pattern.findall(string)[0]
        score = float(score_pattern.findall(string)[0])
        
        return explanation, score

    # def gen_seed_function_prompt(self, seed_thought_prompt):
    #     # combine seed_thought_prompt and function
    #     return seed_thought_prompt + "Implement the function you just described into python code." + self.function_signature
    
    def gen_seed_function_prompt_with_thought(self, thought):
        # combine seed_thought_prompt and function
        return self.sys_prompt + self.game_rules + "You previously came up with the following thoughts and pseudocode: \n" + thought + "\n \n Implement the function you just described into python code. \n" + self.function_signature 
    
    def gen_seed_function_prompt_without_thought(self):
        # combine seed_thought_prompt and function
        return self.sys_prompt + self.game_rules + "Given the rules of the game, come up with a function that can be used to evaluate the value of a state in the game." + self.function_signature

    def gen_feedback_analyze_prompt(self, function, feedback):
        '''
        tells the LLM to analyze the feedback
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        return self.sys_prompt + self.game_rules + PREVIOUS_FUNCTION_INTRO + function + FEEDBACK_INTRO + feedback + CONCLUSION_FROM_FEEDBACK 

    def gen_improvement_thought_prompt(self, previous, prev_thoughts):
        '''
        tells the LLM to improve upon previous thoughts
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        return previous + MODIFY_ABSTRACT + prev_thoughts 

    def gen_specific_improvement_prompt(self, previous_prompt, conclusions, num_ideas = 1):
        '''
        tells the LLM to improve upon previous thoughts
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        return previous_prompt + conclusions + f'''\n \n Based on the function, feedback, and conclusions you drew, what are {num_ideas} improvements that you can make to the function that you think will have the most impact? Be as specific and concrete as possible, mentioning specific code pieces or helper functions that you can add. If the game has different phases, you can also mention what specific phase the idea applies to. Write them out in the following format:
        
        
        Thoughts: <your thoughts here>

        Idea 1: <your idea here>

        Idea 2: <your idea here>

        ...

        Here's an example of what this might look like for 3 improvement ideas:

        Thoughts: The feedback suggests that the main problem is that function is not taking into account the cards in either player's hand, which leads to inaccurate value estimates.

        Idea 1: I can calculate the difference in the number of cards in each player's hand, increasing the estimated value of players with more cards in their hand.

        Idea 2: I can write a helper function that compares the sum of the values of the cards in each player's hand, increasing the estimated value of players with higher total card values.

        Idea 3: During the draw phase, I can also calculate the expected values of the cards that the players will draw, increasing the estimated value of players with higher expected card values.
        '''
    
    @staticmethod
    def parse_improvement_ideas(string: str, num_ideas: int) -> list[str]:
        # Regular expression to find patterns of "Idea X: <idea here>"
        idea_pattern = re.compile(r"Idea \d+: (.+)")
        
        # Find all matches of the pattern in the input string
        matches = idea_pattern.findall(string)
        
        # Return the first num_ideas ideas from the matches
        return matches[:num_ideas]


    def gen_draw_conclusions_from_feedback_prompt(self, function, feedback):
        '''
        tells the LLM to draw conclusions from feedback
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        return self.sys_prompt + self.game_rules + PREVIOUS_FUNCTION_INTRO + function + FEEDBACK_INTRO + feedback + CONCLUSION_FROM_FEEDBACK 

    def gen_implement_function_from_improvement_prompt(self, function, idea):
        '''
        tells the LLM to implement the function from the improvement
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        return self.sys_prompt + self.game_rules + PREVIOUS_FUNCTION_INTRO + function + "\n \n Here is a possible way to improve this function: \n" + idea + "\n \n Implement this improvement into the function as best as you can. Make sure not to change the function signature, which we reproduce below: \n" + self.function_signature

    def gen_improved_function_prompt(self, previous):
        '''
        tells the LLM to generate improved function
        '''
        # combine new_improvements_prompt and new_improvements
        return previous +  '''\n \n Now modify the function you generated previously to improve it based on the feedback given and your thoughts, and output the new function.\n''' + self.function_signature 

    def gen_execution_error_feedback(self, previous, function, execution_error):
        return previous + f"You previously generated the following improved function based on your thoughts and analysis of the feedback: \n {function} \n However, it ran into the following error when running: {execution_error} \n Please fix the error and output your function again." 

    def gen_execution_error_feedback_2(self, error_message):
        string = f'''There was an execution error when running the function you generated on test states. 
        The error message was:
        {error_message}
        Please fix the error and try again.
        '''
        return string

    def gen_single_state_example_feedback(self, i, state_description, estimated_score, intermediate_values, search_score, actual_score): 
        string = f'''---
        Example {i}:
        The state you were trying to estimate a value for is:
        {state_description}

        The function you generated returned the following values:
        {estimated_score}
        for the expected end of game scores of the players. 

        Some intermediate values that you used to calculate the scores were:
        {intermediate_values}

        The estimated end of game scores of the players using lookahead search with your function was:
        {search_score}

        The actual scores of the players at the end of the game in the simulation were:
        {actual_score}
        ---
        '''
        return string
    
    def gen_outcome_single_state_example_feedback(self, i, state_description, estimated_score, intermediate_values, search_score, actual_score): 
        string = f'''---
        Example {i}:
        The state you were trying to estimate a value for is:
        {state_description}

        The function you generated returned the following values:
        {estimated_score}
        for the expected end of game scores of the players. 

        Some intermediate values that you used to calculate the scores were:
        {intermediate_values}

        The actual scores of the players at the end of the game in the simulation were:
        {actual_score}
        ---
        '''
        return string

    @staticmethod
    def gen_state_description(state):
        return str(state)
    
class StrategyPromptGenerator(PromptGenerator):
    '''
    Class that contains all the generation methods for all LLM prompts for self improvement on high level `strategies` (i.e. tips)
    '''

    PREVIOUS_GUIDE_INTRO = '''You previously generated the following section of the strategy guide: \n'''

    def __init__(self, environment_rules, role_name:str):
        function_signature = StrategyPromptGenerator.strategy_guide_signature(role_name)
        sys_prompt = StrategyPromptGenerator.gen_sys_prompt(role_name)
        super().__init__(environment_rules, function_signature, sys_prompt)

    def gen_critic_score_prompt(self, func: str):
        return self.sys_prompt + self.game_rules + "\n Previously you generated the following guide: \n" + func + """\n \n On a scale of 1 to 10, how well do you think this guide will perform in generating dialogue? Please provide a brief explanation for your score. 
        
        You should use the following format for your score and explanation:
        
        Explanation: <your explanation here>
        
        Score: <float, your score here>"""
    
    @staticmethod
    def parse_critic_score(string: str) -> tuple[str, float]:
        # Regular expression to find patterns of "Explanation: <explanation here>"
        explanation_pattern = re.compile(r"Explanation: (.+)")
        
        # Regular expression to find patterns of "Score: <score here>"
        score_pattern = re.compile(r"Score: (\d+\.\d+)")
        
        # Find all matches of the pattern in the input string
        explanation = explanation_pattern.findall(string)[0]
        score = float(score_pattern.findall(string)[0])
        
        return explanation, score

    @staticmethod
    def strategy_guide_signature(role_name: str):
        out = f'''Your guide should be in the form of a worksheet that the student can use to build their speech. You should order the worksheet questions in a way that makes logical sense, and you should have no more than six questions. Your questions should instruct the reader to write parts of their speech. The title of your section should be "Six questions to fill out before speaking as the {role_name} role". Below is an example of how your worksheet should look like:

        Six questions to fill out before speaking as the {role_name} role

        Q1: Which player seems the most suspicious of you and why?

        Q2: For the player that seems the most suspicious of you, produce a statement addressing their suspicious.

        Q3: Which player is the quest leader?

        Q4: Produce a statement addressing the quest leader to convince them to support your intended course of action/ desired team.

        Q5: Which player is the most supportive of you?

        Q6: Produce a statement addressing the supportive player to convince them to support your intended course of action/ desired team.
        '''
        return out
    # @staticmethod
    # def strategy_guide_signature(role_name: str):
    #     out = f'''Your guide should be in the form of a questionaire that the student can use to think through their approach to the current discussion phase. You should order the questions in a way that makes logical sense, and you should have no more than six questions. Try to make your questions as specific as possible. The title of your section should be "Questions to think about to play the {role_name} role effectively during the discussion phase". Below is an example of how your questionaire should look like:

    #     Questions to think about to play the {role_name} role effectively during the discussion phase:

    #     Q1: Which player seems the most suspicious of you and why?

    #     Q2: For the player that seems the most suspicious of you, is it worth convincing them that you are on their side? Why or why not?

    #     Q3: For the player that seems the most suspicious of you, what can you say to convince them that you are on their side?

    #     Q4: What are some conclusions that you can draw from your answers to the previous questions?
    #     '''
    #     return out
    
    # @staticmethod
    # def strategy_guide_signature(role_name: str):
    #     out = f'''Your guide should be in the form of a questionaire that the student can use to think through their approach to the current discussion phase. You should order the questions in a way that makes logical sense, and you should have no more than six questions. Try to make your questions as specific as possible. The title of your section should be "Questions to think about to play the {role_name} role effectively during the discussion phase". Below is an example of how your questionaire should look like:

    #     Questions to think about to play the {role_name} role effectively during the discussion phase:

    #     Q1: Which player seems the most suspicious of you and why?

    #     Q2: For the player that seems the most suspicious of you, is it worth convincing them that you are on their side? Why or why not?

    #     Q3: For the player that seems the most suspicious of you, what can you say to convince them that you are on their side?

    #     Q4: What are some conclusions that you can draw from your answers to the previous questions?

    #     Example of how to fill out this questionaire:

    #     Q1: Which player seems the most suspicious of you and why?
    #     A1: Player 3 seems the most suspicious of me because they have been asking me a lot of questions about my role.

    #     Q2: For the player that seems the most suspicious of you, is it worth convincing them that you are on their side? Why or why not?
    #     A2: It is worth convincing Player 3 that I am on their side because they are a key player in the game and I need their support to win.

    #     Q3: For the player that seems the most suspicious of you, what can you say to convince them that you are on their side?
    #     A3: I can say that I have never been on a failed quest and that I have been trying to reason with the other players to figure out who the Evil players are.

    #     Q4: What are some conclusions that you can draw from your answers to the previous questions?
    #     A4: I should focus on convincing Player 3 that I am on their side and try to get them to support me in the game.
    #     '''
    #     return out

    # @staticmethod
    # def strategy_guide_signature(role_name: str):
    #     out = f'''Your writing should be concise yet informative, providing players with the key strategies and tactics that they can use during the discussion phase in Avalon. You should include examples where necessary to help explain the concepts. Since this is only a section of the guide, it should focus on what things the {role_name} role should say during discussion phase only and should not include information about the game mechanics or the rules of the game.The title of your section should be "How to play the {role_name} role effectively during the discussion phase". 
    #     '''
    #     return out
    
    @staticmethod
    def gen_sys_prompt(role_name: str):
        goal = ""
        # if role_name == "Merlin":
        #     goal = "Specifically, you want to teach the Merlin player how to avoid revealing their identity to the Evil players in this section of the guide.\n"
        # elif role_name == "Assassin" or "Minion":
        #     goal = "Specifically, you want to teach the Evil players how to deceive the Good players into thinking they are Good in this section of the guide.\n"
        # else:
        #     raise ValueError(f"Role name {role_name} not recognized")

        out = f'''You are a coach trying to write a section of a strategy guide on how to play a game well. 
        
        The specific section of the strategy guide you are writing right now is on how to play the {role_name} role effectively during the discussion phase so that they can win the game. Recall that players often use the discussion phase to (1) gather information about other players, (2) try to convince other players of their innocence or guilt, and (3) try to persuade other players of a particular course of action.
        ''' + goal
        return out

    def gen_seed_thought_prompt(self):
        # combine SYS_PROMPT, GOPS_RULES, HEURISTICS_SEED_PROMPT, and GOPS_FUNCTION_SIGNATURE
        return self.sys_prompt + self.game_rules + '''Given the rules of the game, write down your thoughts on how to write the strategy guide. Here's an example:
        
        Thought:
        I should first ask the player to consider which other player is most suspicious of them and why. This will help the player understand the dynamics of the game and how they are perceived by others.
        Next I should ask the player to consider which other player is most supportive of them and why. This will help the player understand who they can trust and rely on in the game.
        '''

    def gen_seed_function_prompt(self, seed_thought_prompt):
        # combine seed_thought_prompt and function
        return seed_thought_prompt + "\n Write out the guide based on your thoughts.\n" + self.function_signature
    
    def gen_seed_function_prompt_with_thought(self, thought):
        # combine seed_thought_prompt and function
        return self.sys_prompt + self.game_rules + "\n You previously came up with the following thoughts: \n" + thought + "\n \n Write out the guide based on your thoughts.\n" + self.function_signature 
    
    def gen_seed_function_prompt_without_thought(self):
        # combine seed_thought_prompt and function
        return self.sys_prompt + self.game_rules + "\n Given the rules of the game, write out the guide.\n" + self.function_signature
    

    
    
    def gen_feedback_analyze_prompt(self, function, feedback):
        '''
        tells the LLM to analyze the feedback
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        out = self.sys_prompt + self.game_rules + self.PREVIOUS_GUIDE_INTRO + function + '\n Below is some feedback on how your guide performed when a student used it to play the game: \n' + feedback + '''\n \n Based on the feedback given and the guide section you generated previously, what are some conclusions you can draw from the feedback? Make sure to cite the specific examples in the feedback to justify your analysis.\n '''

        # self.logger.info(out)
        return out

    def gen_improvement_thought_prompt(self, previous, prev_thoughts):
        '''
        tells the LLM to improve upon previous thoughts
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        out = previous + '''\n \n Given the conclusions you drew from the feedback, modify your previous thoughts accordingly. We reproduce your previous thoughts below for your reference. \n''' + prev_thoughts 
        # self.logger.info(out)
        return out

    # def gen_specific_improvement_prompt(self, previous_prompt, conclusions):
    #     '''
    #     tells the LLM to improve upon previous thoughts
    #     '''
    #     # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
    #     out = previous_prompt + conclusions + '''Based on the function, feedback, and conclusions you drew, what is one improvement that you can make to the guide that you think will have the most impact? Be as specific and concrete as possible.\n '''
    #     self.logger.info(out)
    #     return out

    # def gen_specific_improvement_prompt(self, previous_prompt, conclusions, num_ideas = 1):
    #     '''
    #     tells the LLM to improve upon previous thoughts
    #     '''
    #     # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
    #     return previous_prompt + conclusions + f'''\n \n Based on the questionaire, feedback, and conclusions you drew, what are {num_ideas} improvements that you can make to the questionaire that you think will have the most impact? Be as specific and concrete as possible, including what questions to add, edit, or remove, and write them out in the following format:
        
        
    #     Thoughts: <your thoughts here>

    #     Idea 1: <your idea here>

    #     Idea 2: <your idea here>

    #     ...

    #     Here's an example of what this might look like for 3 improvement ideas:

    #     Thoughts: I should tell the reader to consider how the other players feel about them before speaking in the guide.

    #     Idea 1: Add a question asking the reader to consider which other player is most suspicious of them and why.

    #     Idea 2: Add a question asking the reader to consider which other player is most supportive of them and why.

    #     Idea 3: Add a question asking the reader to consider which other player is most on the fence about them and why.
    #     '''
    
    def gen_specific_improvement_prompt(self, previous_prompt, conclusions, num_ideas = 1):
        '''
        tells the LLM to improve upon previous thoughts
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        return previous_prompt + conclusions + f'''\n \n Based on the worksheet, feedback, and conclusions you drew, what are {num_ideas} improvements that you can make to the worksheet that you think will have the most impact? Be as specific and concrete as possible, including what questions to add, edit, or remove, and write them out in the following format:
        
        
        Thoughts: <your thoughts here>

        Idea 1: <your idea here>

        Idea 2: <your idea here>

        ...

        Here's an example of what this might look like for 3 improvement ideas:

        Thoughts: I should tell the reader to address each player individually in the guide.

        Idea 1: Add a question asking the reader who they think is most suspicious of them and produce a statement addressing their suspicions.

        Idea 2: Add a question asking the reader to consider which other player is most supportive of them and produce a statement addressing their support.

        Idea 3: Add a question asking the reader produce a statement addressed to the quest leader asking them to consider the reader for the quest team. 

        Recall that the goal of Merlin is to convince the Good players that they are Good without revealing that they are Merlin to the Evil players.
        '''
    
    # def gen_specific_improvement_prompt(self, previous_prompt, conclusions, num_ideas = 1):
    #     '''
    #     tells the LLM to improve upon previous thoughts
    #     '''
    #     # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
    #     return previous_prompt + conclusions + f'''\n \n Based on the questionaire, feedback, and conclusions you drew, what are {num_ideas} improvements that you can make to the questionaire that you think will have the most impact? Be as specific and concrete as possible, and write them out in the following format:
        
        
    #     Thoughts: <your thoughts here>

    #     Idea 1: <your idea here>

    #     Idea 2: <your idea here>

    #     ...

    #     Here's an example of what this might look like for 3 improvement ideas:

    #     Thoughts: I should tell the reader to consider how the other players feel about them before speaking in the guide.

    #     Idea 1: I should tell the reader to consider which other player is most suspicious of them and why.

    #     Idea 2: I should tell the reader to consider which other player is most supportive of them and why.

    #     Idea 3: I should tell the reader to consider which other player is most on the fence about them and why.
    #     '''

    def gen_draw_conclusions_from_feedback_prompt(self, function, feedback):
        '''
        tells the LLM to draw conclusions from feedback
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        return self.gen_feedback_analyze_prompt(function, feedback)

    def gen_implement_function_from_improvement_prompt(self, function, idea):
        '''
        tells the LLM to implement the function from the improvement
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        out =  self.sys_prompt + self.game_rules + self.PREVIOUS_GUIDE_INTRO + function + "\n \n Here is a possible way to improve your guide: \n" + idea + "\n\n Implement this improvement into the guide section as best as you can, but do not change the original guide too much. Make sure to stay within the scope of the guide section, which we reiterate below: \n" + self.function_signature
        # self.logger.info(out)
        return out
    
    # def gen_implement_function_thought_from_improvement_prompt(self, function, idea):
    #     '''
    #     tells the LLM to implement the function from the improvement
    #     '''
    #     # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
    #     out =  self.sys_prompt + self.game_rules + self.PREVIOUS_GUIDE_INTRO + function + "\n \n Here is a possible way to improve your guide: \n" + idea + "\n Reflect on how you can use the 
    #     # self.logger.info(out)
    #     return out

    def gen_improved_function_prompt(self, previous):
        '''
        tells the LLM to generate improved function
        '''
        # combine new_improvements_prompt and new_improvements
        out = previous +  '''\n Now modify the guide section you generated previously to improve it based on the feedback given and your thoughts, and output the new section. Do not change the original guide too much. Make sure to stay within the scope of the guide section, which we reiterate below: \n''' + self.function_signature 
        # self.logger.info(out)
        return out

    def gen_execution_error_feedback(self, previous, function, execution_error):
        raise NotImplementedError

    def gen_execution_error_feedback_2(self, error_message):
        raise NotImplementedError

    def gen_single_state_example_feedback(self, i, state_description, estimated_score, intermediate_values, search_score, actual_score): 
        raise NotImplementedError
    
    def gen_strategy_feedback(self, i, dialogue, feedback, eval_player, thought): 
        feedback_str = self.gen_dialogue_feedback_description(feedback)
        string = f'''---
        Example {i}:
        The student using your guide was playing as player {eval_player} in this example filled out your guide as follows:\n 
        {thought}

        Then they said the following during the discussion phase:

        {dialogue}

        Below is some feedback from the other players on how player {eval_player} performed during the discussion phase. Recall that Good players are trying to access which other players are likely to be Good, while Evil players are trying to access which Good player is likely to be Merlin.

        {feedback_str}
        ---
        '''
        return string
    
    def gen_outcome_strategy_feedback(self, i, dialogue, feedback, eval_player, thought, score): 
        # feedback_str = self.gen_dialogue_feedback_description(feedback)
        string = f'''---
        Example {i}:
        The student using your guide was playing as player {eval_player} in this example filled out your guide as follows:\n 
        {thought}

        Then they said the following during the discussion phase:

        {dialogue}

        After being evaluated by the other players, player {eval_player} received a score of {score} for their performance during the discussion phase.
        ---
        '''
        return string
    
    def gen_dialogue_feedback_description(self, feedback: dict[int, tuple[str, str]]):
        feedback_strs = []
        for player, (response, role) in feedback.items():
            feedback_strs.append(f"Player {player} with role {role} had the following thoughts on this discussion round: \n {response}")
        return "\n".join(feedback_strs)