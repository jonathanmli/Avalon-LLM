import yaml
import argparse
# from ....eval import parse_args_to_assignment

# print(sys_args)

parser = argparse.ArgumentParser(description="Argument Parser")

parser.add_argument("--team_discussion", type=bool, default=True, help="Discuss before Team Selection")
parser.add_argument("--test_naive", type=bool, default=False, help="All the agents are naive")

with open("../../../configs/avalon_experiment/args.yaml", 'r') as f:
    content = yaml.safe_load(f)
    print(content)

parser.set_defaults(**content)

args = parser.parse_args()
