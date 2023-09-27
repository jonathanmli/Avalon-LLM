import argparse
import yaml
from ...logger import logger

parser = argparse.ArgumentParser(description="Argument Parser")

parser.add_argument("--task", type=str)
parser.add_argument("--agent", type=str)
parser.add_argument("--config", type=str)

args = parser.parse_args()

print(args.config)

with open(args.config, 'r') as f:
    content = yaml.safe_load(f)
    print(content)

parser.add_argument("--team_discussion", type=bool, default=True, help="Discuss before Team Selection")
parser.add_argument("--test_naive", type=bool, default=False, help="All the agents are naive")
parser.add_argument("--naive_summary", type=str, default="full-history",
                    help="Mode for naive to summarize. Either `full-history` or `10-last`")
parser.add_argument("--agent_summary", type=str, default="full-history",
                    help="Mode for naive to summarize. Either `full-history` or `10-last`")
parser.add_argument("--logging", type=str, default="DEBUG", help="Level of logging info")
parser.add_argument("--thought", type=bool, default=False, help="Append `Think about it and then take actions` to the prompts.")

parser.set_defaults(**content)

args = parser.parse_args()

print(args)

