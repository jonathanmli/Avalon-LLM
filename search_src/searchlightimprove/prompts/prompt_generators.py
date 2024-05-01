from .improvement_prompts import *
class PromptGenerator:
    '''
    Class that contains all the generation methods for all LLM prompts for self improvement

    With all the prompts in one place, it is easier to make changes to the prompts and to keep track of the prompts and prompt tuning
    '''
    def __init__(self, environment_rules, function_signature, sys_prompt = SYS_PROMPT, seed_heuristic_thought_prompt = 0):
        self.game_rules = environment_rules
        self.function_signature = function_signature
        self.sys_prompt = sys_prompt
        self.seed_heuristic_thought_prompt = seed_heuristic_thought_prompt

    def gen_seed_thought_prompt(self):
        # combine SYS_PROMPT, GOPS_RULES, HEURISTICS_SEED_PROMPT, and GOPS_FUNCTION_SIGNATURE
        return self.sys_prompt + self.game_rules + HEURISTICS_SEED_THOUGHT_PROMPTS[self.seed_heuristic_thought_prompt]

    def gen_seed_function_prompt(self, seed_thought_prompt):
        # combine seed_thought_prompt and function
        return seed_thought_prompt + "Implement the function you just described into python code." + self.function_signature
    
    def gen_seed_function_prompt_with_thought(self, thought):
        # combine seed_thought_prompt and function
        return self.sys_prompt + self.game_rules + "You previously came up with the following thoughts and pseudocode: \n" + thought + "Implement the function you just described into python code. \n" + self.function_signature 
    
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

    def gen_specific_improvement_prompt(self, previous_prompt, conclusions):
        '''
        tells the LLM to improve upon previous thoughts
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        return previous_prompt + conclusions + SPECIFIC_IMPROVEMENT_FROM_CONCLUSION

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
        return self.sys_prompt + self.game_rules + PREVIOUS_FUNCTION_INTRO + function + "Here is a possible way to improve this function: \n" + idea + "\n Implement this improvement into the function as best as you can. Make sure not to change the function signature, which we reproduce below: \n" + self.function_signature

    def gen_improved_function_prompt(self, previous):
        '''
        tells the LLM to generate improved function
        '''
        # combine new_improvements_prompt and new_improvements
        return previous +  MODIFY_FUNCTION + self.function_signature 

    def gen_execution_error_feedback(self, previous, function, execution_error):
        return previous + f"You previously generated the following improved function based on your thoughts and analysis of the feedback: \n {function} \n However, it ran into the following error when running: {execution_error} \n Please fix the error and output your function again. Recall that: " + self.function_signature 

    def gen_execution_error_feedback_2(self, error_message):
        string = f'''There was an execution error when running the function you generated on test states. 
        The error message was:
        {error_message}
        Please fix the error and try again.
        '''
        return string

    def gen_single_state_example_feedback(self, i, state_description, estimated_score, intermediate_values, search_score, actual_score): 
        string = f'''--------------------------
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
        --------------------------
        '''
        return string
    
    @staticmethod
    def gen_state_description(state):
        return str(state)