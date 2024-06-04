import random
from game import Game, Move, Player
import typing 


class MinmaxPlayer(Player):
    # NB: a depth of -1 indicates that search will continue until terminal state or quiet
    def __init__(self, player_symbol, heuristic: typing.Callable[[Game], int], max_depth: int = 5):
                    # the input of heuristic function is the State, the output is an int
        super().__init__()
        self.player_symbol = player_symbol
        self.heuristic = heuristic
        self.max_depth = max_depth

    def minimax(self, current_state: Game, depth: int, max_node: bool) -> int:

        if current_state.check_winner != -1 or depth is self.max_depth:
            return self.heuristic(current_state)

        elif max_node:
            best_val = -100000
            actions = current_state.get_available_actions()
            for action in actions:
                current_state.move(action[0], action[1], self.player_symbol)
                value = self.minimax(current_state, depth + 1, False)
                best_val = max(best_val, value)
            return best_val

        else:
            best_val = 100000
            actions = current_state.get_available_actions()
            for action in actions:
                current_state.move(action[0], action[1], self.player_symbol)
                value = self.minimax(current_state, depth + 1, True)
                best_val = min(best_val, value)
            return best_val        
        
    def make_move(self, game) -> tuple[tuple[int, int], Move]:
        action = self.choose_action(game, False)
        return action        

    def choose_action(self, current_state: Game, maximizing_player: bool) -> Move:
        if maximizing_player:
            def func(x, y):
                return x > y
        else:
            def func(x, y):
                return x < y

        actions = current_state.get_available_actions()
        assert (actions is not None)

        best_value = -1000000 if maximizing_player else 1000000

        for action in actions:
            current_state.move(action[0], action[1], self.player_symbol)
            action_value = self.minimax(current_state, 1, not maximizing_player)

            if func(action_value, best_value):
                best_value = action_value
                best_action = action
        return best_action