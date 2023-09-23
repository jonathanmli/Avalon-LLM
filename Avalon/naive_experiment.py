from engine import AvalonGameEnvironment, AvalonConfig
from baseline_agents import *
import random
import argparse
import numpy as np

def gunboat_experiment(agents, config: AvalonConfig, render=False):
    '''
    agents: dictionary of classes of agents
    config: AvalonConfig object
    render: whether to render the game as text
    '''
    # initialize environment
    env = AvalonGameEnvironment(config)
    player_list = []

    if render:
        print(env.get_roles())
    
    player_sides = [side for _, _, side in env.get_roles()]

    for i, (role_i, role_name, side) in enumerate(env.get_roles()):
        player_list.append(agents[role_i](i, f"Player {i}", config))

        # Merlin and all evil players observe true player sides
        if role_i == 0 or side == 0:
            player_list[i].see_sides(player_sides)

        if render:
            print(f"{player_list[i]} is {role_name}")
            # print divider
    
    if render:
        print("-"*50)
        print(f"{player_list[i]} is {role_name}")
        assert list(map(int, player_list[i].player_sides)) == list(map(int, env.get_partial_sides(i)))

    while not env.done:
        # print phase from env
        phase = env.get_phase()[0]
        if render:
            print(env.get_phase()[1])
        
        # if phase is team selection phase, ask for team
        if phase == 0:
            leader = env.get_quest_leader()
            team = player_list[leader].propose_team(env.turn)
            # print(type(team))
            if render:
                print(f"{player_list[leader]} proposed team {team}")
            env.choose_quest_team(team, leader)
            
        
        # if phase is team voting phase, ask for votes
        elif phase == 1:
            votes = [player_list[i].vote_on_team(env.turn, env.get_current_quest_team()) for i in range(config.num_players)]
            outcome = env.vote_on_team(votes)
            if render:
                print(f"Team votes: {votes}, team outcome: {outcome[2]}")

        # if phase is quest voting phase, ask for votes
        elif phase == 2:
            votes = [player_list[i].vote_on_mission(env.turn, env.get_current_quest_team()) for i in env.get_current_quest_team()]
            outcome = env.vote_on_quest(votes)
            if render:
                print(f"Quest votes: {votes}, mission outcome: {outcome[2]}")
                # print divider
                print("-"*50)

            # all players observe mission outcome
            for player in player_list:
                player.observe_mission(env.get_current_quest_team(), env.turn-1, outcome[3])

        # if phase is assassination phase, ask for assassination
        elif phase == 3:
            assassin = env.get_assassin()
            target = player_list[assassin].assassinate()
            if render:
                print(f"Assassination target: {target}")
            _, _, assassinated = env.choose_assassination_target(assassin, target)
            if render:
                print(f"Assassination outcome: {assassinated}")

    if render:
        # print whether good or evil won
        if env.good_victory:
            print("Good wins!")
        else:
            print("Evil wins!")

    return env.good_victory

def main():

    '''
    Usage: python naive_experiment.py --num_players 5 --num_games 10 --render True --seed 0
    '''

    # use arg parser to get number of players, number of games, render flag, and seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_players", type=int, default=5)
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    num_players = args.num_players
    num_games = args.num_games
    render = args.render
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    config = AvalonConfig(num_players)
    agents = {0: NaiveMerlin, 5: NaiveServant, 6: NaiveMinion, 7:NaiveAssassin}
    wins = 0
    for i in range(num_games):
        wins += gunboat_experiment(agents, config, render=render)
    print(f"Win rate: {wins/num_games}")
            

if __name__ == '__main__':
    main()