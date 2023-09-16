import random
from .engine import AvalonConfig

class Agent:

    def __init__(self, id, name, config: AvalonConfig, side=None, role=None, sides = None):
        self.name = name
        self.id = id
        self.config = config
        self.role = role
        self.team = None
        self.side = side # 1 for good, 0 for evil
        self.num_players = config.num_players
        self.history = None
        if sides is None:
            self.player_sides = [-1] * self.num_players # -1 for unknown, 0 for evil, 1 for good
            self.player_sides[id] = side
        else:
            self.player_sides = sides

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    def propose_team(self, mission_id):
        return random.sample(range(0, self.num_players), self.config.num_players_for_quest[mission_id])
    
    def vote_on_team(self, mission_id, team):
        return random.choice([0, 1])
    
    def vote_on_mission(self, mission_id, team):
        return self.side
    
    def assign_side(self, side):
        self.side = side

    def assign_role(self, role):
        self.role = role

    def see_sides(self, sides):
        self.player_sides = sides

    def assignHistory(self, history):
        self.history = history

    def observeMission(self, team, mission_id, num_fails):
        pass

    def assassinate(self):
        pass

class NaiveMinion(Agent):
    
    def __init__(self, id, name, config: AvalonConfig, side=0, role=6, sides = None):
        super().__init__(id, name, config, side, role, sides)

    def vote_on_mission(self, mission_id, team):
        num_fails = self.config.num_fails_required[mission_id]

        # if less than num_fails evil players on the team, vote success
        if sum([self.player_sides[i] == 0 for i in team]) < num_fails:
            return 1
        # if equal to num_fails evil players on the team, vote fail
        elif sum([self.player_sides[i] == 0 for i in team]) == num_fails:
            return 0
        # if there are only two evil players in the game, the assassin is on the team, and num_fails == 1, vote success. otherwise, vote fail
        if self.side == 0 and 7 in team and self.config.num_evil == 2 and num_fails == 1:
            return 1
        else:
            return 0
        
    def vote_on_team(self, mission_id, team):
        # approve if there are at least x evil player(s) on the team, where x is number of fails required for this mission
        num_fails = self.config.num_fails_required[mission_id]
        if sum([self.player_sides[i] == 0 for i in team]) >= num_fails:
            return 1
        else:
            return 0
        
    def propose_team(self, mission_id):
        num_fails = self.config.num_fails_required[mission_id]
        # choose evil team with x-1 other evil player(s), where x is number of fails required for this mission, plus the minion
        evil_team = random.sample([i for i in range(self.num_players) if self.player_sides[i] == 0 and i != self.id], num_fails - 1) + [self.id]

        # propose a random team that includes evil_team and y-x good player(s), where y is number of players required for this mission
        return random.sample([i for i in range(self.num_players) if i not in evil_team and self.player_sides[i] == 1], self.config.num_players_for_quest[mission_id] - num_fails) + evil_team
        
class NaiveAssassin(Agent):
    
    def __init__(self, id, name, config: AvalonConfig, side=0, role=7, sides = None):
        super().__init__(id, name, config, side, role, sides)

    def vote_on_mission(self, mission_id, team):
        # always fail
        return 0
    
    def vote_on_mission(self, mission_id, team):
        num_fails = self.config.num_fails_required[mission_id]

        # if less than num_fails evil players on the team, vote success
        if sum([self.player_sides[i] == 0 for i in team]) < num_fails:
            return 1
        # if equal to num_fails evil players on the team, vote fail
        elif sum([self.player_sides[i] == 0 for i in team]) == num_fails:
            return 0
        # else vote fail
        else:
            return 0
        
    def vote_on_team(self, mission_id, team):
        # approve if there are at least x evil player(s) on the team, where x is number of fails required for this mission
        num_fails = self.config.num_fails_required[mission_id]
        if sum([self.player_sides[i] == 0 for i in team]) >= num_fails:
            return 1
        else:
            return 0
        
    def propose_team(self, mission_id):
        num_fails = self.config.num_fails_required[mission_id]
        # choose evil team with x-1 other evil player(s), where x is number of fails required for this mission, plus the assassin
        evil_team = random.sample([i for i in range(self.num_players) if self.player_sides[i] == 0 and i != self.id], num_fails - 1) + [self.id]

        # propose a random team that includes evil_team and y-x good player(s), where y is number of players required for this mission
        return random.sample([i for i in range(self.num_players) if i not in evil_team and self.player_sides[i] == 1], self.config.num_players_for_quest[mission_id] - num_fails) + evil_team
        
    
class NaiveMerlin(Agent):

    def __init__(self, id, name, config: AvalonConfig, side=1, role=0, sides = None):
        super().__init__(id, name, config, side, role, sides)
    
    def vote_on_team(self, mission_id, team):
        # approve if there are no evil players on the team
        if any([self.player_sides[i] == 0 for i in team]):
            return 0
        else:
            return 1
        
    def propose_team(self, mission_id):
        # propose a random team with all good players that includes Merlin
        return random.sample([i for i in range(self.num_players) if self.player_sides[i] != 0 and i != self.id], self.config.num_players_for_quest[mission_id] - 1) + [self.id]

class NaiveServant(Agent):

    def __init__(self, id, name, config: AvalonConfig, side=1, role=5, sides = None):
        super().__init__(id, name, config, side, role, sides)

        # maintain a list of all possible combinations of player sides
        self.possible_player_sides = self.generate_possible_player_sides(self.sides)

        
    def generate_possible_player_sides(self, sides):
        out = []
        # if there are no unknown sides, return the list of sides   
        if -1 not in sides:
            return [sides]
        else:
            # find the first unknown side
            unknown_index = sides.index(-1)
            # recurse on the two possible sides
            for side in [0, 1]:
                sides_copy = sides.copy()
                sides_copy[unknown_index] = side
                out.extend(self.generate_possible_player_sides(sides_copy))
            return out

        
        

        


    


    
