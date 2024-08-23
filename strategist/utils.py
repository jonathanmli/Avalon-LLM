import heapq
import logging
import datetime
import os
import re
import numpy as np


def setup_logging_environment(log_directory='logs', log_level=logging.DEBUG):
    """
    Sets up the logging environment by ensuring the log directory exists and configuring
    the root logger to use a FileHandler with a unique filename. This setup affects all
    loggers created in the application.

    :param log_directory: The directory where log files will be stored.
    :param log_level: The logging level for the handler.
    """

    # Ensure the logs directory exists
    os.makedirs(log_directory, exist_ok=True)

    # Create a unique filename with the current date and time
    filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
    full_log_path = os.path.join(log_directory, filename)

    # Configure the root logger to use a FileHandler with the unique filename
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        handlers=[logging.FileHandler(full_log_path)])

def configure_class_logger(log_file_path, logger):
    """
    Configures a logger for a class with a FileHandler pointing to a specified log file path.
    This setup includes ensuring the directory exists, setting up the file handler,
    and applying a consistent log message format.

    :param log_file_path: Full path to the log file.
    :param logger: The logger instance to configure.
    """
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Create a file handler for logging to a file
    file_handler = logging.FileHandler(log_file_path)
    
    # Set a formatter for the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    # Optional: Prevent log messages from being propagated to the root logger
    logger.propagate = False

def filter_log_entries(log_file_path, logger_name_to_filter):
    """
    Filters and prints multi-line log entries from a specified log file based on a logger name.
    
    :param log_file_path: Path to the log file to be filtered.
    :param logger_name_to_filter: Logger name to filter the log entries by.

    # Example usage
    log_file_path = "path/to/your/logfile.log"
    logger_name_to_filter = "YourLoggerNameHere"
    filter_log_entries(log_file_path, logger_name_to_filter)
    """
    def is_new_entry(line):
        # Adjust the regex according to the actual timestamp format in your log entries
        return re.match(r'\d{4}-\d{2}-\d{2}', line) is not None

    with open(log_file_path, 'r') as log_file:
        buffer = ""  # Buffer to hold multi-line log entries
        include_buffer = False  # Whether to include the buffered log entry in the output
        
        for line in log_file:
            if is_new_entry(line):
                # When we reach a new log entry, decide whether to print the buffered entry
                if include_buffer:
                    print(buffer, end='')
                
                # Reset buffer and include flag for the new log entry
                buffer = line
                include_buffer = logger_name_to_filter in line
            else:
                # If not a new entry, continue buffering lines
                buffer += line
        
        # Print the last buffered entry if it matches the filter
        if include_buffer:
            print(buffer, end='')



# class UpdatablePriorityDictionary:
#     '''
#     A priority dictionary that allows for updating the priority of an existing key
#     '''

#     def __init__(self):
#         self.heap = []
#         self.entry_finder = {}  # Maps keys to entries
#         self.REMOVED = '<removed-key>'  # Placeholder for a removed key
#         self.counter = 0  # Unique sequence count

#     def add_or_update_key(self, key, value, priority):
#         if key in self.entry_finder:
#             self.remove_key(key)
#         count = self.counter
#         entry = (priority, count, key, value)
#         self.counter += 1
#         self.entry_finder[key] = entry
#         heapq.heappush(self.heap, entry)

#     def remove_key(self, key):
#         entry = self.entry_finder.pop(key)
#         new_entry = (entry[0], entry[1], self.REMOVED, entry[3])
#         self.entry_finder[key] = new_entry

#     def pop_key(self):
#         while self.heap:
#             priority, count, key, value = heapq.heappop(self.heap)
#             if key is not self.REMOVED:
#                 del self.entry_finder[key]
#                 return key, value, priority
#         raise KeyError('pop from an empty priority queue')

#     def get_item(self, key):
#         entry = self.entry_finder.get(key, None)
#         if entry and entry[2] != self.REMOVED:
#             return entry[2], entry[3], entry[0]  # key, value, priority
#         return None
    
#     def get_value(self, key):
#         entry = self.entry_finder.get(key, None)
#         if entry and entry[2] != self.REMOVED:
#             return entry[3]
#         return None

#     def get_items(self):
#         # Returns an iterable over items (key, value, priority) for all not removed entries
#         return [(entry[2], entry[3], entry[0])  # key, value, priority
#                 for entry in self.entry_finder.values() if entry[2] != self.REMOVED]

#     def get_top_k_items(self, k):
#         '''
#         Returns the top k items with the highest priority, sorted from highest to lowest

#         Args:
#             k: number of items to return. if k = int('inf'), return all items sorted

#         Returns:
#             list of tuples: (key, value, priority)
#         '''
#         valid_entries = [entry for entry in self.heap if entry[2] != self.REMOVED]
#         if k == -1:
#             sorted_entries = sorted(valid_entries, key=lambda x: x[0], reverse=True)
#             return [(entry[2], entry[3], entry[0]) for entry in sorted_entries]  # key, value, priority
#         else:
#             top_k_entries = heapq.nlargest(k, valid_entries) # note that this is already sorted
#             return [(entry[2], entry[3], entry[0]) for entry in top_k_entries]  # key, value, priority
        
    
        
# Example usage
# upd_priority_dict = UpdatablePriorityDictionary()
# upd_priority_dict.add_or_update_key('task1', 'Task One Description', 5)
# upd_priority_dict.add_or_update_key('task2', 'Task Two Description', 10)
# upd_priority_dict.add_or_update_key('task3', 'Task Three Description', 2)
# upd_priority_dict.add_or_update_key('task1', 'Task One Updated', 3)  # Update task1

# # Get a specific item
# print("Specific item:", upd_priority_dict.get_item('task1'))

# # Get all items
# print("All items:", upd_priority_dict.get_items())

# # Get top 2 items
# print("Top 2 items:", upd_priority_dict.get_top_k_items(2))
        
# class UPDwithSampling(UpdatablePriorityDictionary):
#     '''
#     An extension of UpdatablePriorityDictionary that allows for sampling an item with probability that is a function of its priority and visit count
#     '''
#     def __init__(self, rng: np.random.Generator = np.random.default_rng()):
#         super().__init__()
#         self.rng = rng

#     def softmax_sample(self, k, temperature):
#         top_k_items = self.get_top_k_items(k)
#         priorities = [item[2] for item in top_k_items]  # Extract priorities

#         # Compute softmax probabilities
#         exp_priorities = np.exp(np.array(priorities) / temperature)
#         probabilities = exp_priorities / exp_priorities.sum()

#         # Sample one of the top k items based on the softmax probabilities
#         sampled_index = self.rng.choice(len(top_k_items), p=probabilities)
#         return top_k_items[sampled_index]


