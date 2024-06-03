from collections import defaultdict
import random
import numpy as np
from copy import deepcopy
from randomplayer import RandomPlayer
from cleverrandomplayer import CleverRandomPlayer
from qlearningplayer import QLearningPlayer
from minmaxplayer import MinimaxPlayer
from montecarloplayer import MonteCarloPlayer
from heuristic import heuristic
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from game import Game, Move, Player

def game(player1, player2):
    quixo_game = Game()
    winner = quixo_game.play(player1, player2)
    return winner

def give_reward(state, player1, player2, action, game):
    if isinstance(player1, QLearningPlayer):
        if game.check_winner() == player1.player_symbol:
            fr = player1.feed_reward(state, 1, action)
        elif game.check_winner() == player2.player_symbol:
            fr = player1.feed_reward(state, -1, action)
        else:
            fr = player1.feed_reward(state, -0.01, action)
    elif isinstance(player2, QLearningPlayer):
        if game.check_winner() == player1.player_symbol:
            fr = player2.feed_reward(state, -1, action)
        elif game.check_winner() == player2.player_symbol:
            fr = player2.feed_reward(state, 1, action)
        else:
            fr = player2.feed_reward(state, -0.01, action)       
    else:
        fr = 0 
    return fr

def train(player1, player2, matches, q_table_file):
    wins = {player1: 0, player2: 0}

    for _ in tqdm(range(matches), desc='Training Progress'):
        quixo_game = Game()
        trajectory = []
        winner = -1

        while winner < 0:
            state = quixo_game.get_board()
            current_player = quixo_game.get_current_player()
            player = player1 if current_player == 1 else player2
            action = player.make_move(quixo_game)
            quixo_game.move(action[0], action[1], current_player)
            trajectory.append((state, action))
            winner = quixo_game.check_winner()

            reward = give_reward(trajectory[-1][0], player1, player2, action, quixo_game)
            for (state, action) in trajectory:
                if isinstance(player1, QLearningPlayer):
                    player1.update_q_value(state, action, reward, trajectory[-1][0])
                elif isinstance(player2, QLearningPlayer):
                    player2.update_q_value(state, action, reward, trajectory[-1][0])

            quixo_game.current_player_idx = 1 if quixo_game.current_player_idx == 0 else 0

        if winner == player1.player_symbol:
            wins[player1] += 1
        elif winner == player2.player_symbol:
            wins[player2] += 1       
        player1.player_symbol, player2.player_symbol = player2.player_symbol, player1.player_symbol

    win_percentage_player1 = (wins[player1] / matches) * 100
    win_percentage_player2 = (wins[player2] / matches) * 100
    print(f'Player 1 {player1.__class__.__name__} symbol {player1.player_symbol} win percentage: {win_percentage_player1}%')
    print(f'Player 2 {player2.__class__.__name__} symbol {player2.player_symbol} win percentage: {win_percentage_player2}%')

    if isinstance(player1, QLearningPlayer):
        player1.save_q_table(q_table_file)
    if isinstance(player2, QLearningPlayer):
        player2.save_q_table(q_table_file)

    return player1.__class__.__name__, player1.player_symbol, win_percentage_player1, player2.__class__.__name__, player2.player_symbol, win_percentage_player2

def test(player1, player2, matches, q_table_file):
    if isinstance(player1, QLearningPlayer) and q_table_file:
        player1.load_q_table(q_table_file)
    if isinstance(player2, QLearningPlayer) and q_table_file:
        player2.load_q_table(q_table_file)

    wins = {player1: 0, player2: 0}
    for _ in tqdm(range(matches), desc='Testing Progress'):

        quixo_game = Game()
        winner = -1
        while winner < 0:
            state = quixo_game._board
            current_player = quixo_game.get_current_player()
            player = player1 if current_player == 1 else player2
            action = player.make_move(quixo_game)
            quixo_game.move(action[0], action[1], current_player)
            winner = quixo_game.check_winner()
            quixo_game.current_player_idx = 1 if quixo_game.current_player_idx == 0 else 0

        if winner == player1.player_symbol:
            wins[player1] += 1
        elif winner == player2.player_symbol:
            wins[player2] += 1

    win_percentage_player1 = (wins[player1] / matches) * 100
    win_percentage_player2 = (wins[player2] / matches) * 100
    print(f'Player 1 {player1.__class__.__name__} symbol {player1.player_symbol} win percentage: {win_percentage_player1}%')
    print(f'Player 2 {player2.__class__.__name__} symbol {player2.player_symbol} win percentage: {win_percentage_player2}%')
    return player1.__class__.__name__, player1.player_symbol, win_percentage_player1, player2.__class__.__name__, player2.player_symbol, win_percentage_player2


def play_game(player1, player2, matches):
    wins = {player1: 0, player2: 0}
    for _ in tqdm(range(matches), desc='Gaming Progress'):

        quixo_game = Game()
        winner = -1
        while winner < 0:
            state = quixo_game._board
            current_player = quixo_game.get_current_player()
            player = player1 if current_player == 1 else player2
            action = player.make_move(quixo_game)
            quixo_game.move(action[0], action[1], current_player)
            winner = quixo_game.check_winner()
            quixo_game.current_player_idx = 1 if quixo_game.current_player_idx == 0 else 0

        if winner == player1.player_symbol:
            wins[player1] += 1
        elif winner == player2.player_symbol:
            wins[player2] += 1

    win_percentage_player1 = (wins[player1] / matches) * 100
    win_percentage_player2 = (wins[player2] / matches) * 100
    print(f'Player 1 {player1.__class__.__name__} symbol {player1.player_symbol} win percentage: {win_percentage_player1}%')
    print(f'Player 2 {player2.__class__.__name__} symbol {player2.player_symbol} win percentage: {win_percentage_player2}%')
    return player1.__class__.__name__, player1.player_symbol, win_percentage_player1, player2.__class__.__name__, player2.player_symbol, win_percentage_player2

def plot(df):
    plt.figure(figsize=(14, 6))
    sns.set(style="whitegrid")
    barplot = sns.barplot(x='Player 1 Name', y='Player 1 Win Percentage', hue='Player 2 Name', data=df, palette='viridis')
    plt.title('Player 1 Win Percentages During Training')
    plt.ylim(0, 110)
    plt.ylabel('Win Percentage (%)')
    plt.xlabel('Player 1 Name (symbol 0)')
    plt.legend(title='Player 2 Name (symbol 1)', loc='center left', bbox_to_anchor=(1, 0.5))
    for p in barplot.patches:
        if p.get_height() > 0:
            barplot.annotate(format(p.get_height(), '.1f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center',
                            xytext = (0, 9),
                            textcoords = 'offset points',
                            fontsize=10, color='black')
    plt.tight_layout()
    plt.show()
    plt.savefig("res.png")

def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        plt.annotate('{}'.format(round(height, 2)),
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), 
                     textcoords="offset points",
                     ha='center', va='bottom')

def plot_q(dftrain, dftest):
    train_results = dftrain[dftrain['Player 1 Name'] == 'QLearningPlayer']
    test_results = dftest[dftest['Player 1 Name'] == 'QLearningPlayer']

    plot_data = {
        'Opponent': ['Random Player', 'Clever Random Player', 'Minimax Player', 'MonteCarlo Player'],
        'Train': list(train_results['Player 1 Win Percentage']),
        'Test': list(test_results['Player 1 Win Percentage'])
    }

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(plot_df))

    bars1 = plt.bar(index, plot_df['Train'], bar_width, label='Train')
    bars2 = plt.bar([i + bar_width for i in index], plot_df['Test'], bar_width, label='Test')

    plt.xlabel('Opponents')
    plt.ylabel('Win Percentage (%)')
    plt.title('QLearningPlayer (symbol 0) playing as 1st vs All Other Players')
    plt.xticks([i + bar_width / 2 for i in index], plot_df['Opponent'])
    plt.legend()

    plt.yticks(np.arange(0, 101, 10))  

    annotate_bars(bars1)
    annotate_bars(bars2)

    plt.show()
    plt.savefig("res_q.png")

def plot_q2(dftrain, dftest):
    train_results = dftrain[dftrain['Player 2 Name'] == 'QLearningPlayer']
    test_results = dftest[dftest['Player 2 Name'] == 'QLearningPlayer']

    plot_data = {
        'Opponent': ['Random Player', 'Clever Random Player', 'Minimax Player', 'MonteCarlo Player'],
        'Train': list(train_results['Player 2 Win Percentage']),
        'Test': list(test_results['Player 2 Win Percentage'])
    }

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(plot_df))

    bars1 = plt.bar(index, plot_df['Train'], bar_width, label='Train')
    bars2 = plt.bar([i + bar_width for i in index], plot_df['Test'], bar_width, label='Test')

    plt.xlabel('Opponents')
    plt.ylabel('Win Percentage (%)')
    plt.title('QLearningPlayer (symbol 1) playing as 2nd vs All Other Players')
    plt.xticks([i + bar_width / 2 for i in index], plot_df['Opponent'])
    plt.legend()

    plt.yticks(np.arange(0, 101, 10))  

    annotate_bars(bars1)
    annotate_bars(bars2)

    plt.show()
    plt.savefig("res_q.png")

if __name__ == "__main__":
    results = []
    results2 = []
    game_results = []
    tresults = []
    tresults2 = []
    players1 = [RandomPlayer(player_symbol=0), CleverRandomPlayer(player_symbol=0), MinimaxPlayer(player_symbol=0, heuristic=heuristic), MonteCarloPlayer(player_symbol=0)]
    players2 = [RandomPlayer(player_symbol=1), CleverRandomPlayer(player_symbol=1), MinimaxPlayer(player_symbol=1, heuristic=heuristic), MonteCarloPlayer(player_symbol=1)]

    print("playing games ...")
    for player1_ in players1:
        for player2_ in players2:
            # if player1_.__class__.__name__ != player2_.__class__.__name__:
                player1_name, player1_symbol, win_percentage_player1, player2_name, player2_symbol, win_percentage_player2 = play_game(player1_, player2_, matches=100)
                game_results.append({
                    'Player 1 Name': player1_name,
                    'Player 1 Symbol': player1_symbol,
                    'Player 1 Win Percentage': win_percentage_player1,
                    'Player 2 Name': player2_name,
                    'Player 2 Symbol': player2_symbol,
                    'Player 2 Win Percentage': win_percentage_player2
                })
    df_res = pd.DataFrame(game_results)

    print("\n")
    player1 =  QLearningPlayer(player_symbol=0, epsilon=0.1, epsilon_decay=0.95, epsilon_min=0.01, alpha=0.2, gamma=0.9)
    for player2 in players2:
        if player1.__class__.__name__ != player2.__class__.__name__:
            player1_name, player1_symbol, win_percentage_player1, player2_name, player2_symbol, win_percentage_player2 = train(player1, player2, matches=100, q_table_file='q_table.pkl')
            test_player1_name, test_player1_symbol, test_win_percentage_player1, test_player2_name, test_player2_symbol, test_win_percentage_player2 = test(player1, player2, matches=50, q_table_file='q_table.pkl')
            results.append({
                'Player 1 Name': player1_name,
                'Player 1 Symbol': player1_symbol,
                'Player 1 Win Percentage': win_percentage_player1,
                'Player 2 Name': player2_name,
                'Player 2 Symbol': player2_symbol,
                'Player 2 Win Percentage': win_percentage_player2
            })
            tresults.append({
                'Player 1 Name': test_player1_name,
                'Player 1 Symbol': test_player1_symbol,
                'Player 1 Win Percentage': test_win_percentage_player1,
                'Player 2 Name': test_player2_name,
                'Player 2 Symbol': test_player2_symbol,
                'Player 2 Win Percentage': test_win_percentage_player2
            })

    dftrain = pd.DataFrame(results)
    dftest = pd.DataFrame(tresults)
    
    print("\n")
    player_2 =  QLearningPlayer(player_symbol=1, epsilon=0.1, epsilon_decay=0.95, epsilon_min=0.01, alpha=0.2, gamma=0.9)
    for player_1 in players1:
        if player1.__class__.__name__ != player2.__class__.__name__:
            player1_name, player1_symbol, win_percentage_player1, player2_name, player2_symbol, win_percentage_player2 = train(player_1, player_2, matches=100, q_table_file='q2_table.pkl')
            test_player1_name, test_player1_symbol, test_win_percentage_player1, test_player2_name, test_player2_symbol, test_win_percentage_player2 = test(player_1, player_2, matches=50, q_table_file='q2_table.pkl')
            results2.append({
                'Player 1 Name': player1_name,
                'Player 1 Symbol': player1_symbol,
                'Player 1 Win Percentage': win_percentage_player1,
                'Player 2 Name': player2_name,
                'Player 2 Symbol': player2_symbol,
                'Player 2 Win Percentage': win_percentage_player2
            })
            tresults2.append({
                'Player 1 Name': test_player1_name,
                'Player 1 Symbol': test_player1_symbol,
                'Player 1 Win Percentage': test_win_percentage_player1,
                'Player 2 Name': test_player2_name,
                'Player 2 Symbol': test_player2_symbol,
                'Player 2 Win Percentage': test_win_percentage_player2
            })

    dftrain2 = pd.DataFrame(results2)
    dftest2 = pd.DataFrame(tresults2)

    plot(df_res)
    plot_q(dftrain, dftest)
    plot_q2(dftrain2, dftest2)
