import random
from game import Game, Move, Player
from heuristic import heuristic
from copy import deepcopy, copy
import math

class MonteCarloPlayer(Player):
    def __init__(self, player_symbol):
        super().__init__()
        self.player_symbol = player_symbol

    # UCB = average_score + C * sqrt(ln(total_simulations) / simulations)
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        num_simulations = 50 
        num_selected_moves = min(len(game.get_available_actions()), 40)
        best_move = None
        best_score = -1000000

        total_simulations = sum(num_simulations for _ in game.get_available_actions())

        for pos, slide in random.sample(game.get_available_actions(), num_selected_moves):
            simulations = num_simulations
            average_score = heuristic(game)
            C = 1  # Trade-off between exploration and exploitation
            ucb_score = average_score + C * math.sqrt(math.log(total_simulations) / simulations)

            total_score = ucb_score
            for _ in range(num_simulations):
                cloned_game = copy(game)
                cloned_game.move(pos, slide, game.get_current_player())

                while cloned_game.check_winner() == -1:
                    valid_move = False
                    while not valid_move:
                        from_pos = (random.randint(0, 4), random.randint(0, 4))
                        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
                        valid_move = game.move(from_pos, move, self.player_symbol)

                if cloned_game.check_winner() == self.player_symbol:
                    total_score += 1
                else:
                    total_score -= 1

            if ucb_score > best_score:
                best_score = ucb_score
                best_move = (pos, slide)

        return best_move    
    