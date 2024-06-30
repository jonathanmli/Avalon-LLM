import numpy as np
import itertools
import heapq
from typing import Any
import logging
from abc import ABC

def dict_to_set_of_cartesian_products(d: dict):
    '''
    Converts a dictionary to a set of cartesian products

    Args:
        d: dictionary from key to set of values

    Returns:
        set_of_cartesian_products: set of cartesian products. 
        each cartesian product is a tuple of tuples (key, value)
        i.e. an element of the returned set would look like ((key1, value1), (key2, value2), ...)

    TODO: there's a but here where if the dictionary is empty, it returns a set with an empty tuple
    '''
     # Generate all combinations of values for each key
    all_combinations = list(itertools.product(*[[(key, value) for value in values] for key, values in d.items()]))
    
    # Convert list of combinations to a set
    set_of_cartesian_products = set(all_combinations)
    return set_of_cartesian_products

class AbstractLogged(ABC):
    _instance_counter = 0

    def __init__(self):
        # Increment the instance counter and set it as part of the logger name
        type(self)._instance_counter += 1
        self.instance_id = type(self)._instance_counter
        
        # Create a logger with a name based on the class name and instance counter
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.instance_id}")

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)
        

class UpdatablePriorityDictionary(AbstractLogged):
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.REMOVED = '<removed-key>'
        self.counter = 0
        super().__init__()

    def add_or_update_key(self, key, value, priority):
        # assert that priority is float like
        assert isinstance(priority, (int, float)), "Priority must be an int or float"
        if key in self.entry_finder:
            self.remove_key(key)
        count = self.counter
        entry = (priority, count, key, value)
        self.counter += 1
        self.entry_finder[key] = entry
        heapq.heappush(self.heap, entry)

    def remove_key(self, key):
        entry = self.entry_finder.pop(key, None)
        if entry:
            new_entry = (float('inf'), entry[1], self.REMOVED, entry[3])
            self.heap.append(new_entry)
            self.heapify_heap()

    def pop_key(self):
        self.clean_heap()
        if not self.heap:
            raise KeyError("pop from an empty priority queue")
        priority, count, key, value = heapq.heappop(self.heap)
        del self.entry_finder[key]
        return key, value, priority

    def get_value(self, key):
        entry = self.entry_finder.get(key, None)
        if entry and entry[2] != self.REMOVED:
            return entry[3]
        return None
    
    def get_item(self, key):
        entry = self.entry_finder.get(key, None)
        if entry and entry[2] != self.REMOVED:
            return entry[2], entry[3], entry[0]  # key, value, priority
        return None

    def get_items(self):
        # Returns an iterable over items (key, value, priority) for all not removed entries
        return [(entry[2], entry[3], entry[0])  # key, value, priority
                for entry in self.entry_finder.values() if entry[2] != self.REMOVED]

    def get_top_k_items(self, k=-1):
        self.clean_heap()
        if k == -1 or k >= len(self.heap):
            sorted_entries = sorted(self.heap, key=lambda x: -x[0])
        else:
            sorted_entries = heapq.nlargest(k, self.heap, key=lambda x: x[0])
        return [(entry[2], entry[3], entry[0]) for entry in sorted_entries]

    def clean_heap(self):
        self.heap = [entry for entry in self.heap if self.entry_finder.get(entry[2], None) == entry]
        self.heapify_heap()

    def heapify_heap(self):
        heapq.heapify(self.heap)
        
    
        
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
        
class UPDwithSampling(UpdatablePriorityDictionary):
    '''
    An extension of UpdatablePriorityDictionary that allows for sampling an item with probability that is a function of its priority and visit count
    '''
    def __init__(self, rng: np.random.Generator = np.random.default_rng()):
        super().__init__()
        self.rng = rng

    def softmax_sample(self, k = -1, temperature = 1.0) -> tuple[Any, Any, float]:
        top_k_items = self.get_top_k_items(k)
        priorities = [item[2] for item in top_k_items]  # Extract priorities

        # log the priorities
        self.logger.debug(f"Priorities: {priorities}")

        # Compute softmax probabilities
        exp_priorities = np.exp(np.array(priorities) / temperature)

        # set any NaNs to 0
        exp_priorities = np.nan_to_num(exp_priorities)

        probabilities = exp_priorities / exp_priorities.sum()

        self.logger.debug(f"exp_priorities: {exp_priorities}")
        self.logger.debug(f"Probabilities: {probabilities}")


        # Sample one of the top k items based on the softmax probabilities
        sampled_index = self.rng.choice(len(top_k_items), p=probabilities)
        return top_k_items[sampled_index]