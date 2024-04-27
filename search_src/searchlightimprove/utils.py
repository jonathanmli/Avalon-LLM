import re
from typing import List

def parse_improvement_list(gpt_response: str) -> List[str]:

    # Split the text based on the pattern "number.)"
    improvements = re.split(r'\d+\.\)', gpt_response)[1:]  # Ignore the first element as it will be empty

    # Strip whitespace from each improvement
    improvements = [improvement.strip() for improvement in improvements]

    return improvements