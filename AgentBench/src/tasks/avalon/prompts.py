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
                            # "Discussion Phase. Please choose 3 players from player ids 0 to 4.",
                            # """Thought: Based on my observation, Player 1 is my team-mate. I'm Player 2 but I'm from the Evil side, it's necessary for me to be in the team. Thus, I will choose [0, 1, 3] as the team.
                            # Statement: choose([0, 2, 3]).""",
                            "Discussion Phase. Please vote on the team [1, 3, 4].",
                            """Thought: I believe one of my Good team-mates (player 4) is in the team. Thus, I approve the team. Also, since I'm from the Good side, I want to let my team-mates know, so I would claim that I'm Good during discussion.
                            Statement: I am Good and I believe one of my team-mates is in the team. I hope Good players to reveal your identities so that we can make better decisions.""",
                            # "Discussion Phase. Please vote on the quest.",
                            # "Thought: Because I'm from the Good side, I approve the quest.",
                            "That's all about the tutorial.",
                            "OKay, I understand."
                            ]
ONE_SHOT_ASSASSIN_DISCUSSION = ["Tutorial on making statements during discussion phase.",
                            "Okay, please start.",
                            # "Please choose 3 players from player ids 0 to 4.",
                            # "Thought: I don't have any observations yet, thus I will choose [0, 2, 3] as the team.\nAction: choose([0, 2, 3]).",
                            "Discussion Phase. Please vote on the team [1, 3, 4].",
                            """Thought: I'm Player 1. I'm from the Evil side, so I will approve the team so that I can fail the quest.
                            Statement: I will approve the team. I'm from the Good side by the way.""",
                            # "Discussion Phase. Please vote on the quest.",
                            # "Thought: Because I'm from the Good side, I approve the quest.",
                            "That's all about the tutorial.",
                            "OKay, I understand."
                            ]
ONE_SHOT_EVIL_DISCUSSION = ["Tutorial on making statements during discussion phase.",
                            "Okay, please start.",
                            # "Please choose 3 players from player ids 0 to 4.",
                            # "Thought: I don't have any observations yet, thus I will choose [0, 2, 3] as the team.\nAction: choose([0, 2, 3]).",
                            "Discussion Phase. Please vote on the team [1, 3, 4].",
                            """Thought: I believe one of my Evil team-mates (player 3) is in the team. However, I'm Player 0 and I'm not in the team. To pretend I'm from the Good side, I would reject the team.
                            Statement: I'm Player 0 and I from the Good side. To ensure the mission is successful, I need to be in the team. Thus, I'm going to reject the team.""",
                            # "Discussion Phase. Please vote on the quest.",
                            # """Thought: Because I'm from the Evil side, I approve the quest. But I'm gonna make some troubles.
                            # Statement: I'm from the Good side and I'm going to approve the quest.""",
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
    'Minion of Modred': ["""Tutorial on strategies:
                        
                        As you are playing the role of Minion of Modred in this game, here are some aspects you can consider when formulating strategies for making decisions.

                        1. Identity Declaration: You can pretend to be on the Good side and influence the Good players to make incorrect decisions.
                         
                        2. Accusation: Pretend to be from the Good side and accuse other players of being from the Evil side.

                        3. Defense: When accused of being from the Evil side, insist that you are actually from the Good side.
                        """],
    'Loyal Servant of Arthur\'s': ["""Tutorial on strategies:
                      
                      As you are playing the role of Loyal Servant of Arthur\'s in this game, here are some aspects you can consider when formulating strategies for making decisions.

                      1. Identity Declaration: You can choose to reveal your true identity to inform players on the Good side. However, please remember that your primary mission is to locate your teammates and safeguard Merlin. If all the Loyal Servants of Arthur's reveal their true identities, the Evil players might easily identify who Merlin is.

                      2. Accusation: You can accuse players you suspect are Evil directly.

                      3. Defense: When accused, you can pretend to be Merlin.
                      """]

}

REVEAL_PROMPTS = {
    "Merlin": [
        "Player {} is Minion of Modred",
        "Players {} are Minions of Modred."
    ],
    "Minion of Modred": [
        "Player {} is Minion of Modred",
        "Players {} are Minions of Modred."
    ],
    # "Percival": "Player {} is Merlin and player {} is Morgan."
}