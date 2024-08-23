from .headers import *
from .llm_utils.llm_api_models import GPT35Multi
# from .prompts.improvement_prompts import *
from .prompts.prompt_generators import PromptGenerator
from strategist.GOPS.baseline_models_GOPS import LLMFunctionalValueHeuristic

class LLMImprovementProposer(ImprovementProposer):

    def __init__(self, LLMmodel: GPT35Multi, prompt_list, check_function: Callable[[Any], bool], prompt_generator: PromptGenerator, regenerate=True):
        '''
        Args:
            LLMmodel: LLM model to use for generating responses
            prompt_list: list of prompts to use for generating responses
            regenerate: whether to regenerate the responses if they are not executable
        
        '''
        self.LLMmodel = LLMmodel
        self.prompt_list = prompt_list
        self.regenerate = regenerate
        super().__init__()
        self.check_function = check_function
        self.prompt_generator = prompt_generator

    def propose(self, base: str, abstract: str = '', feedback: str = '') -> tuple[list[str], list[str]]:
        # TODO: add reflection prompt on what conclusions we can draw from the feedback
        proposed_functions = []
        proposed_abstracts = []

        prompt1 = self.prompt_generator.gen_feedback_analyze_prompt(base, feedback)
        # gen_improvement_thought_prompt(base, feedback, abstract)

        # self.logger.info(f'Prompt for thoughts: {prompt1}')

        # generate responses
        responses1 = self.LLMmodel.generate(prompt1)

        for response1 in responses1:
            prompt2 = self.prompt_generator.gen_improvement_thought_prompt(prompt1+response1, abstract)

            # self.logger.info(f'Prompt for improvements: {prompt2}')

            # generate responses
            responses2 = self.LLMmodel.generate(prompt2)

            for response2 in responses2:
                prompt3 = self.prompt_generator.gen_improved_function_prompt(prompt2 + response2)

                self.logger.info(f'Prompt for improved function: {prompt3}')

                # generate responses
                improved_functions = self.LLMmodel.generate(prompt3)

                # check if responses are executable
                if self.regenerate:
                    for i, improved_function in enumerate(improved_functions):
                        is_executable = False
                        while not is_executable:
                            try:
                                is_executable = self.check_function(improved_function)
                            except Exception as e:
                                self.logger.info(f'Error executing function: {improved_function}, {e}')
                                is_executable = False
                                new_prompt = self.prompt_generator.gen_execution_error_feedback(prompt2 + response2, improved_function, e)
                                improved_function = self.LLMmodel.generate(new_prompt, 1)[0]
                        
                        improved_functions[i] = improved_function

                # append all responses to proposed functions
                proposed_functions.extend(improved_functions)

                for improved_function in improved_functions:
                    self.logger.info(f'Improved function: {improved_function}')
                    proposed_abstracts.append(response2)
                
        return proposed_functions, proposed_abstracts

# Example usage:
# Initialize the GPT35 object with a different temperature and request 3 responses
# gpt = GPT35Multi(temperature=0.7, num_responses=1)
# proposer = LLMImprovementProposer(gpt, ['function to improve:', 'improve function:', 'improve the function:'])
# responses = proposer.propose('def f(x):\n    return x', {})
# print(responses)
