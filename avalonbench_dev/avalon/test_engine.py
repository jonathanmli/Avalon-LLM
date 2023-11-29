from engine import AvalonGameEnvironment, AvalonBasicConfig

def main():
    # ask for number of players
    num_players = int(input("Enter number of players: "))
    config = AvalonBasicConfig(num_players)
    env = AvalonGameEnvironment(config)
    
    # print roles from env
    print(env.get_roles())

    while not env.done:
        # print phase from env
        phase = env.get_phase()[0]
        print(env.get_phase()[1])
        
        # if phase is team selection phase, ask for team
        if phase == 0:
            print(f"Please choose {env.get_team_size()} players in this round.")
            leader = env.get_quest_leader()
            team = [int(x) for x in input(f"Enter team player {leader}: ").split()]
            env.choose_quest_team(team, leader)
        
        # if phase is team voting phase, ask for votes
        elif phase == 1:
            votes = [int(x) for x in input("Enter team votes: ").split()]
            env.vote_on_team(votes)

        # if phase is quest voting phase, ask for votes
        elif phase == 2:
            votes = [int(x) for x in input("Enter quest votes: ").split()]
            env.vote_on_quest(votes)

        # if phase is assassination phase, ask for assassination
        elif phase == 3:
            assassin = env.get_assassin()
            target = int(input(f"Enter assassination target: "))
            env.choose_assassination_target(assassin, target)

    # print whether good or evil won
    if env.good_victory:
        print("Good wins!")
    else:
        print("Evil wins!")
            

if __name__ == '__main__':
    main()