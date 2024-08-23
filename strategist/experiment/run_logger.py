import logging
import os
import datetime

from strategist.searchlightimprove.headers import *
from strategist.searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from strategist.searchlightimprove.proposers import LLMImprovementProposer
from strategist.searchlightimprove.prompts.improvement_prompts import IMPROVEMENT_PROMPTS
from strategist.GOPS.baseline_models_GOPS import *
from strategist.GOPS.value_heuristic_evaluators import GOPSValueHeuristicsSSGEvaluator
from strategist.searchlightimprove.analyzers import HeuristicsAnalyzer
from strategist.searchlight.gameplay.simulators import GameSimulator
from strategist.GOPS.examples.abstract_list3 import abstract_list
from strategist.GOPS.examples.func_list3 import func_list



from strategist.utils import setup_logging_environment

# # Ensure the logs directory exists
# os.makedirs('logs', exist_ok=True)

# # Create a unique filename with the current date and time
# filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
# log_file_path = os.path.join('logs', filename)

# # Configure logging
# logging.basicConfig(filename=log_file_path, level=logging.DEBUG, 
#                     format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


if __name__ == '__main__':
    setup_logging_environment()

    # Log a test message
    logging.info('This is a test log message.')

# print(f"Logging to {log_file_path}")


