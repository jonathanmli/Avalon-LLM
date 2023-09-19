from engine import AvalonGameEnvironment, AvalonConfig
from baseline_agents import *
import random

def main():
    # ask for number of players
    num_players = 5
    # int(input("Enter number of players: "))
    config = AvalonConfig(num_players)
    env = AvalonGameEnvironment(config)
    player_list = []

    # for i in range(num_players):
    #     player_list.append(player(f"Player {i}", num_players, random.choice([0, 1])))

    print(env.get_roles())
    player_sides = [side for _, _, side in env.get_roles()]

    for i, (role_i, role_name, side) in enumerate(env.get_roles()):
        if role_i == 0: # add naive merlin
            player_list.append(NaiveMerlin(i, f"Player {i}", config, sides=player_sides))
        elif role_i == 5: # add naive servant
            player_list.append(NaiveServant(i, f"Player {i}", config))
        elif role_i == 7: # add naive assassin
            player_list.append(NaiveAssassin(i, f"Player {i}", config, sides=player_sides))
        elif role_i == 6: # add naive minion
            player_list.append(NaiveMinion(i, f"Player {i}", config, sides=player_sides))

        print(f"{player_list[i]} is {role_name}")

    while not env.done:
        # print phase from env
        phase = env.get_phase()[0]
        print(env.get_phase()[1])
        
        # if phase is team selection phase, ask for team
        if phase == 0:
            leader = env.get_quest_leader()
            team = player_list[leader].propose_team(env.turn)
            print(f"Please choose {env.get_team_size()} players in this round.")
            print(f"{player_list[leader]} proposed team {team}")
            env.choose_quest_team(team, leader)
            
        
        # if phase is team voting phase, ask for votes
        elif phase == 1:
            votes = [player_list[i].vote_on_team(env.turn, env.get_current_quest_team()) for i in range(num_players)]
            outcome = env.vote_on_team(votes)
            print(f"Team votes: {votes}, team outcome: {outcome[2]}")

        # if phase is quest voting phase, ask for votes
        elif phase == 2:
            votes = [player_list[i].vote_on_mission(env.turn, env.get_current_quest_team()) for i in env.get_current_quest_team()]
            outcome = env.vote_on_quest(votes)
            print(f"Quest votes: {votes}, mission outcome: {outcome[2]}")

            # all players observe mission outcome
            for player in player_list:
                print(env.turn-1)
                player.observe_mission(env.get_current_quest_team(), env.turn-1, outcome[2])

        # if phase is assassination phase, ask for assassination
        elif phase == 3:
            assassin = env.get_assassin()
            target = assassin.assassinate()
            # target = int(input(f"Enter assassination target: "))
            print(f"Assassination target: {target}")
            _, _, assassinated = env.choose_assassination_target(assassin, target)
            print(f"Assassination outcome: {assassinated}")

    # print whether good or evil won
    if env.good_victory:
        print("Good wins!")
    else:
        print("Evil wins!")
            

if __name__ == '__main__':
    main()