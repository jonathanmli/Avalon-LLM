import argparse
import yaml

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

parser.set_defaults(**content)

args = parser.parse_args()

print(args.test_naive)

