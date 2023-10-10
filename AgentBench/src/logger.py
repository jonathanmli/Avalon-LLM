import logging
import os
# from .tasks.avalon.arguments import args
# from .tasks.avalon import logger
logger = logging.getLogger('avalon_logger')

# Check if the folder exists
if not os.path.exists("./src/tasks/avalon/logs/"):
    # If it doesn't exist, create it
    os.makedirs("./src/tasks/avalon/logs/")

counter = os.listdir('./src/tasks/avalon/logs/')

# Create a file handler
file_handler = logging.FileHandler(f'./src/tasks/avalon/logs/{len(counter)}.log')

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)