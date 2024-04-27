import logging
import os
import datetime

from src.searchlightimprove.headers import *
from src.searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from src.searchlightimprove.proposers import LLMImprovementProposer
from src.searchlightimprove.prompts.improvement_prompts import IMPROVEMENT_PROMPTS
from src.GOPS.baseline_models_GOPS import *
from src.GOPS.value_heuristic_evaluators import GOPSValueHeuristicsSSGEvaluator
from src.searchlightimprove.analyzers import HeuristicsAnalyzer
from src.searchlight.gameplay.simulators import GameSimulator
from src.GOPS.examples.abstract_list3 import abstract_list
from src.GOPS.examples.func_list3 import func_list



from src.utils import setup_logging_environment

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


