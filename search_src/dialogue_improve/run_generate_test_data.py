from .data_loader import DataLoader

FILE_PATH = 'src/dialogue_improve/test_data.json'

DISCUSSION_HISTORY = "Start of game. \n"

DIALOGUE = [ "Greetings, fellow players. As a Servant of Arthur, my primary goal is to ensure the success of our Quests. I am committed to working together with all of you to achieve victory for Good.\n I believe that for this first Quest, a team consisting of Player 1 and Player 4 would be a strong choice. I hope for your support in approving this team, as I am confident that we can work effectively together to accomplish our goal.\nI encourage everyone to share their thoughts and strategies openly. Let's work together to assess each other's intentions and make the best decisions for the success of our Quest. Remember, the fate of Camelot rests in our hands, and we must be vigilant against the forces of Evil.", "Greetings, noble companions. I echo Player 1's sentiment in working together for the success of our Quests. I believe it is essential to carefully consider our team choices to ensure victory for Good.\nI suggest that for this first Quest, a team consisting of Player 1 and Player 2 would provide a strong foundation for success. Together, we can overcome any challenges that come our way.\nLet us unite in our efforts and show our unwavering dedication to the cause. I urge all players to carefully evaluate each team proposal and remember that unity is key to thwarting the forces of Evil.", "Greetings, noble companions. I echo the sentiments shared by Player 1 and Player 2 regarding our shared goal of ensuring the success of our Quests. Unity and cooperation will be key in our endeavors to protect Camelot from the forces of Evil.\n As we deliberate on the composition of our team for this first Quest, I would like to offer a perspective that considers the balance and strength of our choices. A team comprised of Player 1 and Player 3 would bring together a diverse set of skills and viewpoints that can greatly contribute to our success.\n By including Player 3 in the team, we can ensure a comprehensive approach to overcoming any challenges that may arise. Let us work together in harmony, leveraging our individual strengths for the greater good of Camelot. I urge my fellow players to consider this team proposal thoughtfully, as I believe it holds great promise for the success of our Quest.", "Greetings, honorable players. I appreciate the dedication and commitment displayed by Player 1, Player 2, and Player 3 towards our common goal of ensuring the success of this Quest. Unity and cooperation are indeed vital in our quest to safeguard Camelot.\n I believe that the proposed team of Player 1 and Player 2 presents a strong foundation for success. Their combined skills and perspectives can greatly benefit our mission, and I urge all players to consider this team thoughtfully for the betterment of our cause.\n Let us unite our strengths and work together seamlessly to overcome any challenges that may arise. By standing united, we can ensure the triumph of Good over the forces of Evil. I urge all players to support the team of Player 1 and Player 2 for the success of this Quest.", "Greetings, noble companions. I appreciate the thoughtful considerations put forth by Player 1, Player 2, Player 3, and Player 4 in selecting the team for our first Quest. Unity and cooperation are indeed paramount in our quest for victory.\n I find the proposed teams of Player 1 and Player 4, as well as Player 1 and Player 2, to be intriguing choices that certainly possess strong merits. It's essential for us to carefully weigh our options to ensure the success of this Quest.\n In the spirit of fostering diverse perspectives and collaborative efforts, I would like to highlight the potential strengths that a team consisting of Player 1 and Player 0 could bring to our mission. By combining our unique skills and viewpoints, I believe we can pave the way for a successful outcome.\n Let us all approach this decision with open minds and consider the implications of each team composition. Our unity and discernment will be critical in safeguarding Camelot against the forces of Evil. I urge my fellow players to deliberate thoughtfully and choose the team that best aligns with our shared goal of victory."]

SPEAKER_ORDER = [1,2,3,4,0]

ROLES = [0,5,6,5,7]

# quest_leader, phase, turn, round, quest_team, team_votes, quest_results
# STATE_INFO = tuple(1, 0, 0,  0, set(), tuple(), )


# num_players, quest_leader, phase, turn, round, done, good_victory, quest_team, team_votes, quest_votes, quest_results, roles
STATE_TUPLE = tuple([5, 1, 0, 0, 0, False, False, [], tuple(), tuple(), tuple(), tuple(ROLES)])

PRIVATE_INFORMATIONS = ["You are Player 0, with identity Merlin. You are on the side of Good. The Evil players are Players 2 and 4. Please do not forget your identity throughout the game.", "You are Player 1, with identity Servant of Arthur. You are on the side of Good. Please do not forget your identity throughout the game.", "You are Player 2, with identity Minion of Mordred. You are on the side of Evil. Players 2 and 4 are Evil. Please do not forget your identity throughout the game.", "You are Player 3, with identity Servant of Arthur. You are on the side of Good. Please do not forget your identity throughout the game.", "You are Player 4, with identity Assassin. You are on the side of Evil. Players 2 and 4 are Evil. Please do not forget your identity throughout the game."]

ACTION_INTENTS = [{0,1}, {1,2}, {1,2}, {1,3}, {1,2}]
ACTION_INTENTS = [list(intent) for intent in ACTION_INTENTS] # NOTE: json only handles lists, not sets

def main():
    data_loader = DataLoader()
    data_loader.add_data_point(DISCUSSION_HISTORY, STATE_TUPLE, ACTION_INTENTS, PRIVATE_INFORMATIONS, ROLES, DIALOGUE, SPEAKER_ORDER)
    data_loader.save_data(FILE_PATH)

    # try loading the data
    data_loader = DataLoader()
    data_loader.load_data(FILE_PATH)

    # try sampling the data
    discussion_history, state_info, intended_actions, private_informations, roles, dialogue, speaking_order = data_loader.sample_data_point()

    print(discussion_history)
    print(state_info)
    print(intended_actions)
    print(private_informations)
    print(roles)
    print(dialogue)
    print(speaking_order)

if __name__ == '__main__':
    main()