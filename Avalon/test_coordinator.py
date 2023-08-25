from engine import AvalonGameEnvironment
import random


class player:

    def __init__(self, name, num_players, side=None):
        self.name = name
        self.num_players = num_players
        self.role = None
        self.team = None
        self.side = side # 1 for good, 0 for evil

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    def propose_team(self, team_size):
        return random.sample(range(0, self.num_players), team_size)
    
    def vote_on_team(self, team):
        return random.choice([0, 1])
    
    def vote_on_mission(self):
        return self.side
    
    def assign_side(self, side):
        self.side = side

    def assign_role(self, role):
        self.role = role

    def assassinate(self, player):
        if player >= self.num_players:
            raise ValueError(f"Player {player} does not exist.")
        if self.role != 7:
            raise ValueError("Only assassin can assassinate.")
        return random.randint(0, self.num_players-1)



def main():
    # ask for number of players
    num_players = int(input("Enter number of players: "))
    env = AvalonGameEnvironment(num_players)
    player_list = []

    for i in range(num_players):
        player_list.append(player(f"Player {i}", num_players, random.choice([0, 1])))

    print(env.get_roles())

    for i, (role_i, role_name, side) in enumerate(env.get_roles()):
        player_list[i].assign_role(role_i)
        player_list[i].assign_side(side)
        print(f"{player_list[i]} is {role_name}")

    while not env.done:
        # print phase from env
        phase = env.get_phase()[0]
        print(env.get_phase()[1])
        
        # if phase is team selection phase, ask for team
        if phase == 0:
            leader = env.get_quest_leader()
            team = player_list[leader].propose_team(env.get_team_size())
            print(f"Please choose {env.get_team_size()} players in this round.")
            env.choose_quest_team(team, leader)
            print(f"{player_list[leader]} proposed team {team}")
        
        # if phase is team voting phase, ask for votes
        elif phase == 1:
            votes = [player_list[i].vote_on_team(env.get_current_quest_team()) for i in range(num_players)]
            outcome = env.vote_on_team(votes)
            print(f"Team votes: {votes}, team outcome: {outcome[2]}")

        # if phase is quest voting phase, ask for votes
        elif phase == 2:
            votes = [player_list[i].vote_on_mission() for i in env.get_current_quest_team()]
            outcome = env.vote_on_quest(votes)
            print(f"Quest votes: {votes}, mission outcome: {outcome[2]}")

        # if phase is assassination phase, ask for assassination
        elif phase == 3:
            assassin = env.get_assassin()
            target = assassin.assassinate()
            # target = int(input(f"Enter assassination target: "))
            print(f"Assassination target: {target}")
            env.choose_assassination_target(assassin, target)

    # print whether good or evil won
    if env.good_victory:
        print("Good wins!")
    else:
        print("Evil wins!")
            

if __name__ == '__main__':
    main()