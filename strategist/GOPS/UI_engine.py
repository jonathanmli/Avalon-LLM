from engine import GOPSConfig, GOPSEnvironment

def main():
    # simple text based UI for playing GOPS with 2 players

    num_turns = int(input("Enter number of turns: "))

    config = GOPSConfig(num_turns)
    env = GOPSEnvironment(config)
    (done, score_card, contested_points) = env.reset()

    player1 = input("Enter name for player 1: ")
    player2 = input("Enter name for player 2: ")
    print(f"Welcome {player1} and {player2} to GOPS!")
    while not done:
        print(f"Current score: {player1}: {env.player1_score}, {player2}: {env.player2_score}")
        print(f"Current contested points: {contested_points}, current contested score card: {score_card}")
        move1 = input(f"{player1}, play a card out of {env.player1_hand}: ")
        move2 = input(f"{player2}, play a card out of {env.player2_hand}: ")
        (done, score_card, contested_points) = env.play_cards(int(move1), int(move2))
        print(f"{player1} played {move1}, {player2} played {move2}")
    print(f"Final score: {player1}: {env.player1_score}, {player2}: {env.player2_score}")
    print("Thanks for playing GOPS!")
    
if __name__ == '__main__':
    main()