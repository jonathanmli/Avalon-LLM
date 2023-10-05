INTRODUCTION = """You are playing a game called The Resistance: Avalon.

The Resistance: Avalon is the game of hidden identities and social dedeuction. There are two teams in the game: Good and Evil. Each player has a hidden identity and side. 

There are five Quests in the game. Good players aim to help three Quests succeed, while Evil players aim to fail three Quests. Different quests require different numbers of players to participate. 

Each round, after a round of discussion, the leader will select a team of players to participate in the Quest. Then, all players will vote on whether to approve or reject the team. If the team is approved (a strict majority vote to approve), the Quest will be carried out. If the team is not approved, the next player becomes the leader and the next round will start. If four teams are rejected in a row, the fifth team will automatically be approved.

If the team is approved, each teammember chooses to pass or fail the Quest anonymously. Usually if there is at least one fail vote, the Quest fails. Otherwise, the Quest succeeds.

Below are the roles in the game:

Servant of Arthur (Servant): A good player who does not know who is on the Evil side. Servant's job is to help Good players to complete three Quests without being failed by the Evil players.

Minion of Mordred (Minion): An Evil player who knows who is on the Evil side. Minion's job is to fail three Quests without being identified by the Good players.

Merlin: A Good player who knows who is on the Evil side. Merlin's job is to help Good players to complete three Quests without being failed by the Evil players. However, if the Evil players can identify who Merlin is, the Evil players will win the game immediately.

Assassion: An Evil player who knows who is on the Evil side. Assassin's job is to assassinate Merlin if the Evil players can identify who Merlin is. If Assassin successfully assassinates Merlin, the Evil players will win the game immediately.

Hence, Evil players usually know who is on the Evil side, but Good players usually do not know who is on the Evil side. 

Players may make any claims during the game, at any point in the game. Discussion, deception, accusation, persuasion, and logical deduction are all equally important in order for Good to prevail or Evil to rule the day. Hence, players should rarely reveal their true identity to other players. 
"""

ONE_SHOT_GOOD = ["Tutorial of taking actions by thinking and using tools during action phase.",
                "Okay, please start.",
                "Action Phase. Please choose 3 players from player ids 0 to 4.",
                "Thought: I don't have any observations yet, thus I will choose [0, 2, 3] as the team.\nAction: choose([0, 2, 3]).",
                "Action Phase. Please vote on the team [1, 3, 4].",
                "Thought: I believe one of my Good team-mates (player 4) is in the team. Thus, I approve the team.\nAction: vote(True).",
                "Action Phase. Please vote on the quest.",
                "Thought: Because I'm from the Good side, I approve the quest.\nAction: vote(True).",
                "That's all about the tutorial.",
                "OKay, I understand."
                ]
ONE_SHOT_ASSASSIN = ["Tutorial of taking actions by thinking and using tools during action phase.",
                    "Okay, please start.",
                    "Action Phase. Please choose 3 players from player ids 0 to 4.",
                    "Thought: I don't have any observations yet, thus I will choose [0, 2, 3] as the team.\nAction: choose([0, 2, 3]).",
                    "Action Phase. Please vote on the team [1, 3, 4].",
                    "Thought: I believe one of my Evil team-mates (player 4) is in the team. Thus, I approve the team.\nAction: vote(True).",
                    "Action Phase. Please vote on the quest.",
                    "Thought: Although I'm from the Evil side, I don't want to reveal that in this round. So, I approve the quest.\nAction: vote(True).",
                    "Action Phase. Assassination phase. Your job is to assassinate Merlin. Choose a player (id) to assassinate. Choose the player id from 0 to 9.",
                    "Thought: Based on the observations, I think player 3 is Merlin, so I will assassinate player 3.\nAction: assassinate(3).",
                    "That's all about the tutorial.",                    
                    "OKay, I understand."
                    ]

ONE_SHOT_ASSASSIN_NO_THOUGHT = ["Tutorial of taking actions by thinking and using tools during action phase.",
                    "Okay, please start.",
                    "Please choose 3 players from player ids 0 to 4.",
                    "Action: choose([0, 2, 3]).",
                    "Please vote on the team [1, 3, 4].",
                    "Action: vote(True).",
                    "Please vote on the quest.",
                    "Action: vote(True).",
                    "Assassination phase. Your job is to assassinate Merlin. Choose a player (id) to assassinate. Choose the player id from 0 to 9.",
                    "Action: assassinate(3).",
                    "That's all about the tutorial.",                    
                    "OKay, I understand."
                    ]

ONE_SHOT_EVIL = ["Tutorial of taking actions by thinking and using tools during action phase.",
                    "Okay, please start.",
                    "Action Phase. Please choose 3 players from player ids 0 to 4.",
                    "Thought: I don't have any observations yet, thus I will choose [0, 2, 3] as the team.\nAction: choose([0, 2, 3]).",
                    "Action Phase. Please vote on the team [1, 3, 4].",
                    "Thought: I believe one of my Evil team-mates (player 4) is in the team. Thus, I approve the team.\nAction: vote(True).",
                    "Action Phase. Please vote on the quest.",
                    "Thought: Although I'm from the Evil side, I don't want to reveal that in this round. So, I approve the quest.\nAction: vote(True).",
                    "That's all about the tutorial.",
                    "OKay, I understand."
                    ]

ONE_SHOT_GOOD_DISCUSSION = ["Tutorial on making statements during discussion phase.",
                           "Okay, please start.",
                           "Discussion Phase. Please choose 3 players from player ids 0 to 4.",
                           "Thought: Based on my observation, Player 1 is my team-mate. I'm Player 2 but I'm from the Evil side, it's necessary for me to be in the team. Thus, I will choose [0, 1, 3] as the team. But I want to know what other players think of the team proposal first.\nStatement: Perhaps I will choose [0, 1, 3] as the team.",
                           "Discussion Phase. Please vote on the team [1, 3, 4].",
                           "Thought: I believe one of my Good team-mates (player 4) is in the team. Thus, I approve the team. Also, since I'm from the Good side, I want to let my team-mates know, so I would claim that I'm Good during discussion.\nStatement: I am Good and I believe one of my team-mates is in the team. I hope Good players to reveal your identities so that we can make better decisions.",
                           # "Discussion Phase. Please vote on the quest.",
                           # "Thought: Because I'm from the Good side, I approve the quest.",
                           "Discussion Phase. Please discuss on the assassination target.",
                           "Thought: Since I'm on the Good side and I'm not Merlin, I need to protect Merlin from being identified. Therefore, I can accuse other players of being Merlin and influence Assassin to make correct decisions.\nStatement: Based on my observation, I think Player 3 is Merlin.",
                           "That's all about the tutorial.",
                           "OKay, I understand."
                           ]

ONE_SHOT_MERLIN_DISCUSSION = ["Tutorial on making statements during discussion phase.",
                           "Okay, please start.",
                           "Discussion Phase. Please choose 3 players from player ids 0 to 4.",
                           "Thought: Based on my observation, Player 1 is my team-mate. I'm Player 2 but I'm from the Evil side, it's necessary for me to be in the team. Thus, I will choose [0, 1, 3] as the team. But I want to know what other players think of the team proposal first.\nStatement: Perhaps I will choose [0, 1, 3] as the team.",
                           "Discussion Phase. Please vote on the team [1, 3, 4].",
                           "Thought: I believe one of my Good team-mates (player 4) is in the team. Thus, I approve the team. Also, since I'm from the Good side, I want to let my team-mates know, so I would claim that I'm Good during discussion.\nStatement: I am Good and I believe one of my team-mates is in the team. I hope Good players to reveal your identities so that we can make better decisions.",
                           # "Discussion Phase. Please vote on the quest.",
                           # "Thought: Because I'm from the Good side, I approve the quest.",
                           "Discussion Phase. Please discuss on the assassination target.",
                           "Thought: Since I'm Merlin, I will try not revealing my true identity and pretend to be other characters from the Good side.\nStatement: Based on my observation, I think Player 2 is Merlin.",
                           "That's all about the tutorial.",
                           "OKay, I understand."
                           ]
ONE_SHOT_ASSASSIN_DISCUSSION = ["Tutorial on making statements during discussion phase.",
                           "Okay, please start.",
                           "Discussion Phase. Please choose 3 players from player ids 0 to 4.",
                           "Thought: I don't have any observations yet, thus I will randomly choose [0, 2, 3] as the team. Also, because I'm Assassin, I may aim to deduce who Merlin is during the discussion phase.\nStatement: Maybe I will randomly choose [0, 2, 3] as the team because I don't have any observations yet.",
                           "Discussion Phase. Please vote on the team [1, 3, 4].",
                           "Thought: I'm Player 1. I'm from the Evil side, so I will approve the team so that I can fail the quest.\nStatement: I will approve the team. I'm from the Good side by the way.",
                           # "Discussion Phase. Please vote on the quest.",
                           # "Thought: Because I'm from the Good side, I approve the quest.",
                           "Discussion Phase. Please discuss on the assassination target.",
                           "Tought: Since I'm Assassin, I will directly reveal my identity and see what other players will say.\nStatement: I'm Assassin, and I want to choose the assassination target based on your statements.",
                           "That's all about the tutorial.",
                           "OKay, I understand."
                           ]

ONE_SHOT_EVIL_DISCUSSION = ["Tutorial on making statements during discussion phase.",
                           "Okay, please start.",
                           "Discussion phase. Please choose 3 players from player ids 0 to 4.",
                           "Thought: Although the game just started, I'm Player 2 and I want to include myself in the team and claim that I'm from the Good side to influence the Good players. My current team choice would be [1, 2, 3].\nStatement: I choose [1, 2, 3] as the team. Because as you can see, I'm Player 2 and I'm from the Good side. I want to ensure that the mission in this round is successful. So, please tell me who you are and I can make a better team choice.",
                           "Discussion Phase. Please vote on the team [1, 3, 4].",
                           "Thought: I believe one of my Evil team-mates (player 3) is in the team. However, I'm Player 0 and I'm not in the team. To pretend I'm from the Good side, I would reject the team.\nStatement: I'm Player 0 and I from the Good side. To ensure the mission is successful, I need to be in the team. Thus, I'm going to reject the team.",
                           # "Discussion Phase. Please vote on the quest.",
                           # """Thought: Because I'm from the Evil side, I approve the quest. But I'm gonna make some troubles.
                           # Statement: I'm from the Good side and I'm going to approve the quest.""",
                           "Discussion Phase. Please discuss on the assassination target.",
                           "Thought: Since I'm on the Evil side, my mission is to help Assassin identify who Merlin is. I will provide useful information to my teammate based on my observations.\nStatement: Based on my observation, Player 2 has a high probability to be Merlin.",
                           "That's all about the tutorial.",
                           "OKay, I understand."
                           ]

TUTORIAL_STRATEGIES_PROMPTS_ZERO_SHOT = {
    'Merlin': ["""Tutorial on strategies:

As you are playing the role of Merlin in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: Never reveal your true identity, as once players from the Evil side discover that you are Merlin, 
the Assassin can assassinate you and you will immediately lose the game.

2. Accusation: Exercise caution when accusing players from the Evil side. Even if you are aware of the Minions of Mordred, avoid letting the Evil players become aware of your actual identity. Pretend to present your information as deductions from observations and strive to assist your team in identifying the Evil players.

3. Defense: When other players accuse you of being Merlin, try to defend yourself.""",
               "Okay, I understand"],
    'Minion': ["""Tutorial on strategies:

As you are playing the role of Minion of Modred in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can pretend to be on the Good side and influence the Good players to make incorrect decisions.
    
2. Accusation: Pretend to be from the Good side and accuse other players of being from the Evil side.

3. Defense: When accused of being from the Evil side, insist that you are actually from the Good side.
                        """,
                        "Okay, I understand"],
    'Servant': ["""Tutorial on strategies:

As you are playing the role of Servant in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can choose to reveal your true identity to inform players on the Good side. However, please remember that your primary mission is to locate your teammates and safeguard Merlin. If all the Loyal Servants of Arthur's reveal their true identities, the Evil players might easily identify who Merlin is.

2. Accusation: You can accuse players you suspect are Evil directly.

3. Defense: When accused, you can pretend to be Merlin.
                      """,
                      "Okay, I understand"],
    'Assassin': ["""Tutorial on strategies:

As you are playing the role of Assassin in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can pretend to be from the Good side and influence the decission of the Good players

2. Accusation: You can accuse any players to be on the Evil side to pretend you are Good.

3. Defense: When accused, you can pretend to be from the Good side.
                      """,
                      "Okay, I understand"]

}

REVEAL_PROMPTS = {
    #TODO: Evil, Good
    "Merlin": [
        "Player {} is Minion. Players {} are Servants.",
        # "Players {} are Minions of Modred."
    ],
    #TODO: only knows who are Evil
    "Minion": [
        "Player {} is Assassin. Players {} are Good.",
        "Players {} are Minions of Modred."
    ],
    #TODO: Evil, Good
    "Assassin": [
        "Player {} is Minion. Players {} are Good.",
        "Players {} are Minions of Modred."
    ],
    # "Percival": "Player {} is Merlin and player {} is Morgan."
}

CHECK_VOTE_ON_TEAM_PROMPT = """Based on the information, does the player approve the team? Please answer with the following template:

Answer: {Yes|No}
"""

CHECK_VOTE_ON_QUEST_PROMPT = """Based on the information, does the player support the quest? Please answer with the following template:

Answer: {Yes|No}
"""

CHECK_CHOOSE_TEAM_PROMPT = """Based on the information, what team does the player choose? Please answer with the following template:

Answer: [player_ids]
"""

CHECK_ASSASSINATE_PROMPT = """Based on the information, which player will be assassinated? Please answer with the following template:

Answer: [player_id_num]
"""

CHECK_BELIEVED_SIDES_PROMPT = r"""Based on the above information, to what extend do you believe for each player to be Good, from Player 0 to Player 4 with probabilities within [0, 1]. Please summarize with the following template:

Answer: {0: score_for_0, 1: score_for_1, 2: score_for_2, 3: score_for_3, 4: score_for_4}
"""

COTHOUGHT_PROMPT = """Please think about it step by step and in accordance with your overall strategy, and then take actions."""