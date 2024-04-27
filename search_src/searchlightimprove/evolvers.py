from .headers import *
from src.searchlight.bandit import MultiarmedBanditLearner
from .llm_utils.llm_api_models import LLMModel
# from .prompts.improvement_prompts import gen_specific_improvement_prompt, gen_draw_conclusions_from_feedback_prompt, gen_implement_function_from_improvement_prompt
from .prompts.prompt_generators import PromptGenerator
from src.searchlight.utils import UpdatablePriorityDictionary

import numpy as np
import os

from typing import Optional

class BeamEvolver(Evolver):
    '''
    Abstract class for evolving functions

    TODO: this class is no longer abstract, so we should rename it
    '''

    functions_dict: UpdatablePriorityDictionary # where values are (abstract, feedback, iteration)

    def __init__(self, evaluator: Evaluator, analyzer: FeedbackAnalyzer, prompt_generator: PromptGenerator, batch_size: int = 10, seed_functions: Optional[list[tuple[str, dict]]] = None, check_function: Callable[[str], bool] = lambda x: True, parse_function: Callable[[str],str] = lambda x: x, model: Optional[LLMModel] = None, num_fittest_functions: int = 1):
        '''
        Args:
            evaluator: evaluator for functions
            batch_size: number of functions to propose at a time
            seed_functions: list of seed functions to start with, (function, abstract)
            check_function: function to check if a function is valid
            parse_function: function to parse a function
            model: LLM model to use for generating functions
            num_fittest_functions: number of fittest functions to consider each iteration
        '''
        super().__init__()
        if model is None:
            model = GPT35Multi()
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.analyzer = analyzer
        self.functions_dict = UpdatablePriorityDictionary()
        self.num_evolutions = 0 # number of evolutions conducted
        self.check_function = check_function
        self.model = model
        self.prompt_generator = prompt_generator
        self.parse_function = parse_function
        self.num_fittest_functions = num_fittest_functions

        # create logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # if seed functions are None, generate them
        if seed_functions is None:
            prompt = self.prompt_generator.gen_seed_function_prompt_without_thought()
            functions = self.generate_seed_functions(self.batch_size, prompt)
            seed_functions = [(function, dict()) for function in functions]

        # add seed functions
        self.add_seed_functions(seed_functions)
            
    def add_seed_functions(self, seed_functions: list[tuple[str, dict]]):
        # evaluate and add seed functions
        scores, eval_notes = self.evaluator.evaluate([function for function, abstract in seed_functions])
        # print(scores)
        for i, (function, func_notes) in enumerate(seed_functions):
            # feedback = self.analyzer.translate(eval_notes[i])
            notes = {'feedback': eval_notes[i], 'iteration': 0, 'generation': 0, 'predecessor_function': None} | func_notes
            self.functions_dict.add_or_update_key(function, notes, scores[i]) # TODO: check sign of score

    def generate_seed_functions(self, num_seed_functions: int, prompt: str) -> list[str]:
        '''
        Generates seed functions to start the evolution
        '''
        seed_functions = []
        for _ in range(num_seed_functions):
            seed_function = self.model.generate(prompt, 1)[0]
            seed_function = self.parse_function(seed_function)
            seed_functions.append(seed_function)
        return seed_functions

    def get_fittest(self, k: int = 1) -> list[tuple[str, dict, float]]:
        '''
        Returns the k fittest items (highest to lowest). If there are less than k functions, return all functions

        Items of the form (function, dict(abstract, feedback, iteration), priority)
        '''
        return self.functions_dict.get_top_k_items(k)
    
    def evaluate(self, functions) -> tuple[list[float], list[dict]]:
        return self.evaluator.evaluate(functions)
    
    def evolve_once(self):
        '''
        Conducts one cycle of evolution
        '''

        # get the fittest functions equal to the batch size
        fittest_items = self.get_fittest(self.num_fittest_functions)

        # propose improvements for the fittest functions
        proposed_functions = []
        proposed_generations = []
        predecessor_functions = []
        is_new_function = []
        counter = 0
        while counter < self.batch_size:
            for func, info, priority in fittest_items:
                processed_feedback = self.analyzer.translate(info['feedback'])
                generation = info['generation']

                prompt = self.prompt_generator.gen_draw_conclusions_from_feedback_prompt(func, processed_feedback)
                conclusions = self.model.generate(prompt, 1)[0]
                prompt = self.prompt_generator.gen_improved_function_prompt(prompt + conclusions)
                new_func = self.generate_function(prompt)

                if new_func is None:
                    new_func = func
                    is_new_function.append(False)
                else:
                    is_new_function.append(True)

                proposed_functions.append(new_func)
                proposed_generations.append(generation + 1)
                predecessor_functions.append(func)
                counter += 1

        # evaluate the proposed functions
        scores, notes = self.evaluate(proposed_functions)

        self.num_evolutions += 1

        # log the following things: self.num_evolutions, proportion of non-zero scores, best score
        non_inf_scores = sum(score != float('-inf') for score in scores)
        proportion_non_inf_scores = non_inf_scores / len(scores)
        self.logger.info(f'Evolution: {self.num_evolutions}, Number of genearted functions: {len(proposed_functions)}, Proportion of executable functions: {proportion_non_inf_scores}, Best score: {max(scores)}')

        # add the proposed functions to the dictionary
        for i, function in enumerate(proposed_functions):
            # filter out non-executable functions
            if not is_new_function[i] or scores[i] == float('-inf'):
                continue
            else:
                info = {'feedback': notes[i], 'iteration': self.num_evolutions, 'generation': proposed_generations[i], 'predecessor_function': predecessor_functions[i]}
                self.functions_dict.add_or_update_key(function, info, scores[i])

    def evolve(self, num_cycles: int):
        '''
        Evolves the functions for a certain number of cycles
        '''
        for _ in range(num_cycles):
            self.evolve_once()

    def generate_function(self, prompt: str, tries: int = 4) -> Optional[str]:
        '''
        Generates a function given a prompt

        TODO: generalize this to different temperature, multi-generate
        '''
        new_function = self.model.generate(prompt, 1)[0]
        # parse the function
        new_function = self.parse_function(new_function)

        # check if responses are executable
        is_executable = False
        for i in range(tries):
            try:
                # print check function

                is_executable = self.check_function(new_function)
                return new_function
            except Exception as e:
                self.logger.info(f'Error: \n {e} \n while executing function: \n --- \n  {new_function}')
                is_executable = False
                new_prompt = self.prompt_generator.gen_execution_error_feedback(prompt, new_function, e)
                new_function = self.model.generate(new_prompt, 1)[0]
            # self.logger.info(f'Generated function: {new_function}')
            # i += 1
        
        return None


    def produce_analysis(self, k: int = -1, evaluator: Optional[Evaluator] = None) -> tuple[list[dict], dict]:
        '''
        Produces an analysis of the k fittest functions

        Returns:
            results: dictionary of lists of the results, sorted by final score. includes whatever info is stored in the functions_dict
            benchmark_scores: dictionary of benchmark scores
        '''
        if evaluator is None:
            evaluator = self.evaluator
        fittest_items = self.get_fittest(k)

        # filter out the functions that are not executable
        fittest_items = [(func, info, priority) for func, info, priority in fittest_items if priority != -float('inf')]
        
        # use the evaluator to evaluate the fittest functions with benchmark
        functions = [func for func, _, priority in fittest_items]
        function_scores, function_notes, benchmark_scores = self.evaluator.evaluate_with_benchmark(functions)
        
        # store the results in a list of dictionaries
        results = []
        for i, (func, info, priority) in enumerate(fittest_items):
            # append info dictionary along with the final score and function and estimated score
            to_append = info | {'function': func, 'final_score': function_scores[i], 'estimated_score': priority}
            results.append(to_append)

        # sort results by final score
        results = sorted(results, key=lambda x: x['final_score'], reverse=True)

        # log the estimated scores, final score, generation, and iteration for each function
        for info in results:
            self.logger.info(f'Estimated Score: {info["estimated_score"]}, Final Score: {info["final_score"]}, Iteration: {info["iteration"]}, Generation: {info["generation"]}')

        # log the benchmark scores
        for benchmark_name, benchmark_score in benchmark_scores.items():
            self.logger.info(f'Benchmark {benchmark_name} score: {benchmark_score}')

        # log total number of functions
        self.logger.info(f'Total number of functions: {len(results)}')
        return results, benchmark_scores

    def produce_figures(self, results: list[dict], benchmark_scores: dict, save_directory: str = 'outputs/'):
        '''
        Produces figures from the results
        '''
        # convert the results to a pandas dataframe
        df = pd.DataFrame(results)

        # create a scatter plot of the iteration (x-axis) vs the final score (y-axis)
        # add benchmark scores as horizontal lines
        fig = px.scatter(df, x='iteration', y='final_score', hover_data=['generation'], title='Iteration vs Final Score')
        for benchmark_name, benchmark_score in benchmark_scores.items():
            fig.add_hline(y=benchmark_score, line_dash='dash', annotation_text=f'{benchmark_name} benchmark', annotation_position='top right')

        # create a scatter plot of the iteration (x-axis) vs the estimated score (y-axis)
        # add benchmark scores as horizontal lines
        fig2 = px.scatter(df, x='iteration', y='estimated_score', hover_data=['generation'], title='Iteration vs Estimated Score')
        # for benchmark_name, benchmark_score in benchmark_scores.items():
        #     fig2.add_hline(y=benchmark_score, line_dash='dash', annotation_text=f'{benchmark_name} benchmark', annotation_position='top right')

        # Sort the DataFrame by 'generation' for logical sequencing (if not already sorted)
        df.sort_values('generation', inplace=True)

        # create a scatter plot of the generation (x-axis) vs the final score (y-axis)
        # add benchmark scores as horizontal lines
        fig3 = px.scatter(df, x='generation', y='final_score', hover_data=['iteration'], title='Generation vs Final Score')
        for benchmark_name, benchmark_score in benchmark_scores.items():
            fig3.add_hline(y=benchmark_score, line_dash='dash', annotation_text=f'{benchmark_name} benchmark', annotation_position='top right')

        # Add lines for 'function' to 'predecessor_function' connections
        for index, row in df.iterrows():
            if pd.notna(row['predecessor_function']):  # Check if the predecessor_function exists
                predecessor_row = df[df['function'] == row['predecessor_function']].iloc[0]
                # Draw a line between the current point and its predecessor
                fig3.add_shape(type='line',
                            x0=predecessor_row['generation'], y0=predecessor_row['final_score'],
                            x1=row['generation'], y1=row['final_score'],
                            line=dict(color='RoyalBlue', width=1),
                            )

        # create a scatter plot of the generation (x-axis) vs the estimated score (y-axis)
        fig4 = px.scatter(df, x='generation', y='estimated_score', hover_data=['iteration'], title='Generation vs Estimated Score')

        # Add lines for 'function' to 'predecessor_function' connections
        for index, row in df.iterrows():
            if pd.notna(row['predecessor_function']):  # Check if the predecessor_function exists
                predecessor_row = df[df['function'] == row['predecessor_function']].iloc[0]
                # Draw a line between the current point and its predecessor
                fig4.add_shape(type='line',
                            x0=predecessor_row['generation'], y0=predecessor_row['estimated_score'],
                            x1=row['generation'], y1=row['estimated_score'],
                            line=dict(color='RoyalBlue', width=1),
                            )

        # save the figures to the save directory. include date and time in the filename
        date_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fig.write_html(f'{save_directory}/iteration_vs_final_score_{date_name}.html')
        fig2.write_html(f'{save_directory}/iteration_vs_estimated_score_{date_name}.html')
        fig3.write_html(f'{save_directory}/generation_vs_final_score_{date_name}.html')
        fig4.write_html(f'{save_directory}/generation_vs_estimated_score_{date_name}.html')

        # save df as csv to the save directory. include date and time in the filename
        df.to_csv(f'{save_directory}/results_{date_name}.csv')

class ThoughtBeamEvolver(BeamEvolver):
    '''
    Conducts beam evolution with thought
    '''
    def __init__(self, evaluator: Evaluator, analyzer: FeedbackAnalyzer, prompt_generator: PromptGenerator, batch_size: int = 10, seed_functions: Optional[list[tuple[str, dict]]] = None, check_function: Callable[[str], bool] = lambda x: True, parse_function: Callable[[str],str] = lambda x: x, model: Optional[LLMModel] = None, num_fittest_functions: int = 1):
        
        # generate seed functions with thought if not provided
        if seed_functions is None:
            seed_functions = []
            for i in range(batch_size):
                thought_prompt = self.prompt_generator.gen_seed_thought_prompt()
                thought = self.model.generate(thought_prompt, 1)[0]
                function_prompt = self.prompt_generator.gen_seed_function_prompt_with_thought(thought)
                function_str = self.generate_function(function_prompt)
                if function_str is not None:
                    seed_functions.append((function_str, {'abstract': thought}))

        super().__init__(evaluator=evaluator, analyzer=analyzer, prompt_generator=prompt_generator, batch_size=batch_size, seed_functions=seed_functions, check_function=check_function, parse_function=parse_function, model=model, num_fittest_functions=num_fittest_functions)


    def evolve_once(self):
        '''
        Conducts one cycle of evolution
        '''

        # get the fittest functions equal to the batch size
        fittest_items = self.get_fittest(self.num_fittest_functions)

        # propose improvements for the fittest functions
        proposed_functions = []
        proposed_abstracts = []
        proposed_generations = []
        predecessor_functions = []
        is_new_function = []

        counter = 0
        while counter < self.batch_size:
            for func, info, priority in fittest_items:
                abstract = info['abstract']
                feedback = info['feedback']
                generation = info['generation']

                # first draw conclusions from feedback
                processed_feedback = self.analyzer.translate(feedback)
                prompt = self.prompt_generator.gen_draw_conclusions_from_feedback_prompt(func, processed_feedback)
                conclusions = self.model.generate(prompt, 1)[0]

                # next modify previous abstract with conclusions
                prompt = self.prompt_generator.gen_improvement_thought_prompt(prompt + conclusions, abstract)
                new_abstract = self.model.generate(prompt, 1)[0]

                # generate the new function
                prompt = self.prompt_generator.gen_improved_function_prompt(prompt + new_abstract)
                new_func = self.generate_function(prompt)

                # if new function is not executable, add old function. TODO: this is a hack
                if new_func is None:
                    new_func = func
                    is_new_function.append(False)
                else:
                    is_new_function.append(True)
                proposed_functions.append(new_func)
                proposed_abstracts.append(new_abstract)
                proposed_generations.append(generation + 1)
                predecessor_functions.append(func)
                counter += 1

        # evaluate the proposed functions
        scores, notes = self.evaluate(proposed_functions)

        self.num_evolutions += 1

        # log the following things: self.num_evolutions, proportion of non-zero scores, best score
        non_inf_scores = sum(score != float('-inf') for score in scores)
        proportion_non_inf_scores = non_inf_scores / len(scores)
        self.logger.info(f'Evolution: {self.num_evolutions}, Number of genearted functions: {len(proposed_functions)}, Proportion of executable functions: {proportion_non_inf_scores}, Best score: {max(scores)}')

        # add the proposed functions to the dictionary
        for i, function in enumerate(proposed_functions):
            # filter out non-executable functions
            if not is_new_function[i] or scores[i] == float('-inf'):
                continue
            else:
                info = {'abstract': proposed_abstracts[i], 'feedback': notes[i], 'iteration': self.num_evolutions, 'generation': proposed_generations[i], 'predecessor_function': predecessor_functions[i]}
                # feedback = self.analyzer.translate(notes[i])
                self.functions_dict.add_or_update_key(function, info, scores[i])

class ImprovementLibraryEvolver(BeamEvolver):
    '''
    Conducts evolution with a scored library (bandit learner) of improvement ideas.

    This will build upon the base Evolver, but with an additional library of improvement ideas with scores that will be used to guide the evolution.
    '''
    mbleaner: MultiarmedBanditLearner
    evaluator: Evaluator
    model: LLMModel
    batch_size: int
    analyzer: FeedbackAnalyzer
    prompt_generator: PromptGenerator
    functions_dict: UpdatablePriorityDictionary
    num_evolutions: int


    def __init__(self, evaluator: Evaluator, model: LLMModel, 
                 analyzer: FeedbackAnalyzer, prompt_generator: PromptGenerator, batch_size: int = 10,
                 mbleaner: Optional[MultiarmedBanditLearner] = None,
                 seed_functions: list[tuple[str, dict]] = [], check_function: Callable[[str], bool] = lambda x: True, num_ideas_per_iteration: int = 2, 
                 parse_function: Callable[[str],str] = lambda x: x, num_fittest_functions: int = 1):
        '''
        Args:
            evaluator: evaluator to use for evaluating functions
            model: LLM model to use for generating functions
            analyzer: feedback analyzer to use for analyzing feedback
            batch_size: number of functions to sample from the function library
            mbleaner: bandit learner to use for storing improvement ideas
            seed_functions: seed functions to start the evolution
            check_function: function to check if a function is valid
            implement_steps_per_grow: number of steps to implement per grow
        '''
        super().__init__(evaluator=evaluator, model=model, analyzer=analyzer, batch_size=batch_size, seed_functions=seed_functions, check_function=check_function, prompt_generator=prompt_generator, parse_function=parse_function, num_fittest_functions=num_fittest_functions)
        if mbleaner is None:
            mbleaner = MultiarmedBanditLearner()
        self.mbleaner = mbleaner
        self.num_ideas_per_iteration = num_ideas_per_iteration

        # num implements per iteration should be batch_size integer divided by num_fittest_functions
        self.num_implements_per_iteration = self.batch_size // self.num_fittest_functions
        self.num_idea_loops = self.num_ideas_per_iteration // self.num_fittest_functions

        # if self.batch_size % self.num_fittest_functions is not 0, log a warning
        if self.batch_size % self.num_fittest_functions != 0:
            self.logger.warning(f'Batch size {self.batch_size} is not divisible by num fittest functions {self.num_fittest_functions}')
        # if self.num_ideas_per_iteration % self.num_fittest_functions is not 0, log a warning
        if self.num_ideas_per_iteration % self.num_fittest_functions != 0:
            self.logger.warning(f'Num ideas per iteration {self.num_ideas_per_iteration} is not divisible by num fittest functions {self.num_fittest_functions}')
        

    def add_seed_functions(self, seed_functions: list[tuple[str, str]]) -> None:
        '''
        Adds seed functions to the function library

        Args:
            seed_functions: seed functions to add

        NOTE: we use raw_feedback as the feedback for the seed functions
        '''
        # evaluate and add seed functions
        scores, notes = self.evaluator.evaluate([function for function, abstract in seed_functions])

        # log scores and notes
        self.logger.debug('Seed function scores: %s', scores)
        self.logger.debug('Seed function notes: %s', notes)
    

        for i, (function, abstract) in enumerate(seed_functions):
            info = {'abstract': abstract, 'raw_feedback': notes[i], 'iteration': 0, 'generation': 0, 'idea_trace': [], 'predecessor_function': None}
            self.functions_dict.add_or_update_key(function, info, scores[i]) # NOTE: check sign of score. correct, UPD gets items in descending order

            # log the seed functions
            self.logger.info(f"Seed function {function} with score {scores[i]} and feedback {notes[i]} added to the function library")

    def generate_improvement_ideas(self, batch_size, num_loops:int =1, num_ideas:int =1, improvement_prior=0.0) -> None:
        '''
        Generates improvement ideas and adds them to the bandit learner

        This is basically a reflection step where the agent reflects on the feedback and generates improvement ideas.
        
        We get the top batch_size functions from our function library along with their numerical feedback. We then pass the feedback to the feedback analyzer to sample and translate the feedback to numerical form. We then ask the LLM to reflect on the feedback and generate num_ideas improvement ideas. We then add the improvement ideas to the bandit learner.

        Args:
            batch_size: number of functions to sample from the function library
            num_loops: number of times to repeat the process
            num_ideas: number of improvement ideas to generate per prompt
        '''
        if num_ideas > 1:
            raise NotImplementedError("Generating multiple improvement ideas per prompt is not yet supported")

        # get the top batch_size functions from the function library
        top_items = self.get_fittest(batch_size)

        # extract the functions and feedback from top_items
        functions = []
        unprocessed_feedback = []
        for function, info, score in top_items:
            functions.append(function)
            unprocessed_feedback.append(info['raw_feedback'])

            # log unprocessed feedback
            self.logger.info(f"Function {function} with feedback {info['raw_feedback']} selected for improvement idea generation")

        # sample and translate the feedback
        # processed_feedback = [self.analyzer.translate(data) for data in unprocessed_feedback]

        improvement_ideas = []
        feedback_conclusions = []

        for function, feedback in zip(functions, unprocessed_feedback):
            for i in range(num_loops):
                self.logger.info(f"Generating improvement ideas for function {function} with feedback {feedback}")
                # sample and translate the feedback
                processed_feedback = self.analyzer.translate(feedback)

                # draw conclusions from the feedback
                prompt = self.prompt_generator.gen_draw_conclusions_from_feedback_prompt(function, processed_feedback)
                conclusions = self.model.generate(prompt, 1)[0]
                feedback_conclusions.append(conclusions)

                # generate improvement idea
                prompt = self.prompt_generator.gen_specific_improvement_prompt(prompt, conclusions)
                improvement_idea = self.model.generate(prompt, 1)[0]
                # TODO: diversity would increase if we generated multiple improvement ideas per function

                improvement_ideas.append(improvement_idea)

        # add the improvement ideas to the bandit learner
        scores = [improvement_prior for _ in improvement_ideas]
        for idea, score, conclusion in zip(improvement_ideas, scores, feedback_conclusions):
            self.mbleaner.add_or_update(idea, score, {'feedback_conclusion': conclusion, 'iteration': self.num_evolutions, 'num_implements': 0})
    
    def implement_and_evaluate(self, batch_size) -> None:
        '''
        Samples 1 idea from the bandit learner, applies it to the top batch_size functions, and evaluates the results.

        Adds the new functions to the function library.
        Updates the score of the idea based on how much it improved the functions.

        Args:
            batch_size: number of functions to sample from the function library
        '''
        idea, idea_notes, idea_score = self.mbleaner.softmax_sample()
        top_items = self.get_fittest(batch_size)
        new_functions = []
        is_new_function = []
        prev_scores = [item[2] for item in top_items]
        

        # NOTE: we do not pass the abstracts for now
        for function, info, prev_score in top_items:
            
            prompt = self.prompt_generator.gen_implement_function_from_improvement_prompt(function, idea)

            # generate the new function
            new_function = self.generate_function(prompt)
            # if new function is not executable, add old function. TODO: this is a hack
            if new_function is None:
                new_function = function
                is_new_function.append(False)
            else:
                is_new_function.append(True)
            new_functions.append(new_function)

        # evaluate the new functions
        scores, notes = self.evaluate(new_functions)

        # assert that scores is a list of floats
        assert all(isinstance(score, float) for score in scores)


        # log scores
        self.logger.debug('new_function scores: %s', scores)

        # remove non-new and unexecutable functions from the lists
        new_functions = [new_function for new_function, is_new, score  in zip(new_functions, is_new_function, scores) if is_new and score != float('-inf')]
        scores = [score for score, is_new in zip(scores, is_new_function) if is_new and score != float('-inf')]
        notes = [note for note, is_new, score in zip(notes, is_new_function, scores) if is_new and score != float('-inf')]

        # if there are no new functions, return
        if len(new_functions) == 0:
            self.logger.info(f"No new functions generated from idea {idea}")
            return

        # add new functions to the function library
        for i, (new_function, score, note) in enumerate(zip(new_functions, scores, notes)):
            info = {'abstract': '', 'raw_feedback': note, 'iteration': self.num_evolutions, 'generation': top_items[i][1]['generation'] + 1, 'idea_trace': top_items[i][1]['idea_trace'] + [idea], 'predecessor_function': top_items[i][0]}
            self.functions_dict.add_or_update_key(new_function, info, score)

        # log scores
        self.logger.debug('new new_function scores: %s', scores)

        avg_improvement_score = float(np.mean(scores) - np.mean(prev_scores))

        # debug log np.mean(scores), np.mean(prev_scores), avg_improvement_score
        self.logger.debug(f"Average improvement score: {avg_improvement_score}")
        self.logger.debug(f"Average new function score: {np.mean(scores)}")
        self.logger.debug(f"Average old function score: {np.mean(prev_scores)}")

        # increment the number of implementations of the idea by 1
        idea_notes['num_implements'] += 1

        # update the score of the idea
        self.mbleaner.add_or_update(idea, avg_improvement_score, idea_notes)

        self.logger.info(f"Idea {idea} implemented with average improvement score {avg_improvement_score}")
        

    def evolve_once(self) -> None:
        '''
        Evolves the population once
        '''
        # generate improvement ideas
        self.generate_improvement_ideas(batch_size=self.num_fittest_functions, num_loops=self.num_idea_loops)

        for _ in range(self.num_implements_per_iteration):
            # implement and evaluate
            self.implement_and_evaluate(batch_size=self.num_fittest_functions)

        self.num_evolutions += 1

    def produce_analysis(self, k:int = -1, evaluator: Optional[Evaluator] = None) -> tuple[list, dict, list[dict]]:
        '''
        Produces an analysis of the k fittest functions

        Args:
            k: number of functions to analyze
            evaluator: evaluator to use for evaluating functions for final analysis

        Returns:
            function_results: dictionary of lists of the functions (and info), sorted by final score
            benchmark_scores: dictionary of benchmark scores
            idea_results: dictionary of the ideas (and info) and their scores
        '''
        function_results, benchmark_scores = super().produce_analysis(k, evaluator=evaluator)

        # get the top k ideas
        top_idea_items = self.mbleaner.get_top_k_items(k)

        # store the idea results in a dictionary of lists (to be converted to a pandas dataframe)
        # columns: idea, score
        idea_results = []
        for idea, info, score in top_idea_items:
            to_append = info | {'idea': idea, 'score': score}
            idea_results.append(to_append)

        # log idea results
        self.logger.info(f"Top {k} ideas:")
        for idea_result in idea_results:
            self.logger.info(f"Idea: {idea_result['idea']}, Score: {idea_result['score']}")

        # log total number of ideas
        self.logger.info(f'Total number of ideas: {len(idea_results)}')
        return function_results, benchmark_scores, idea_results

    def produce_figures(self, function_results, benchmark_scores, idea_results, save_directory: str = 'outputs/'):
        '''
        Produces figures from the results
        '''
        super().produce_figures(function_results, benchmark_scores, save_directory=save_directory)

        self.logger.info(f"Saving figures to {save_directory}")

        self.logger.debug(idea_results)

        self.logger.debug(type(idea_results))

        # convert the idea results to a pandas dataframe
        idea_df = pd.DataFrame(idea_results)

        # save the idea results to a csv file with savename 'idea_results.csv'
        idea_df.to_csv(os.path.join(save_directory, 'idea_results.csv'))

        # create a boxplot of the idea scores using plotly
        fig = px.box(idea_df, y='score', title='Idea Scores')
        fig.write_html(os.path.join(save_directory, 'idea_scores_boxplot.html'))

