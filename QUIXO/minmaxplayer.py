import random
from game import Game, Move, Player
import typing 


class MinimaxPlayer(Player):
    # NB: a depth of -1 indicates that search will continue until terminal state or quiet
    def __init__(self, player_symbol, heuristic: typing.Callable[[Game], int], max_depth: int = -1, quiescence_max_depth: int = -1):
                    # the input of heuristic function is the State, the output is an int
        super().__init__()
        self.player_symbol = player_symbol
        self.heuristic = heuristic
        self.max_depth = max_depth
        self.quiescence_max_depth = quiescence_max_depth

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

    def alpha_beta_solve(self, current_state: Game, depth: int, max_node: bool, alpha: int, beta: int) -> int:
        if current_state.check_winner()!= -1:
            return self.heuristic(current_state)
        elif depth is self.max_depth:
            return self.quiescence_search(current_state, 5, max_node, alpha, beta)

        elif max_node:
            best_val = -100000
            actions = current_state.get_available_actions()
            for action in actions:
                current_state.move(action[0], action[1], self.player_symbol)
                value = self.alpha_beta_solve(current_state, depth + 1, False, alpha, beta)
                best_val = max(best_val, value)
                if best_val >= beta:
                    return best_val
                alpha = max(best_val, alpha)
                if alpha >= beta:
                    break                
            return best_val

        else:
            best_val = 100000
            actions = current_state.get_available_actions()
            for action in actions:
                current_state.move(action[0], action[1], self.player_symbol)
                value = self.alpha_beta_solve(current_state, depth + 1, True, alpha, beta)
                best_val = min(best_val, value)
                if best_val <= alpha:
                    return best_val
                beta = min(best_val, beta)
                if alpha >= beta:
                    break                    
            return best_val
     
    def quiescence_search(self, current_state: Game, depth: int, max_node: bool, alpha: int, beta: int) -> int:
        if current_state.check_winner()!= -1 or depth is self.quiescence_max_depth:
            return self.heuristic(current_state)
        non_quiescent_actions = current_state.get_non_quiescent_actions()

        if not non_quiescent_actions:
            # quiet state
            return self.heuristic(current_state)

        elif max_node:
            best_val = -100000
            stand_pat = self.heuristic(current_state)
            if stand_pat >= beta:
                return stand_pat
            if alpha < stand_pat:
                alpha = stand_pat

            for action in non_quiescent_actions:
                current_state.move(action[0], action[1], self.player_symbol)
                value = self.quiescence_search(current_state, depth + 1, False, alpha, beta)
                best_val = max(best_val, value)
                if best_val >= beta:
                    return best_val
                alpha = max(best_val, alpha)
            return best_val

        else:
            stand_pat = self.heuristic(current_state)
            if stand_pat <= alpha:
                return stand_pat
            if stand_pat < beta:
                beta = stand_pat

            best_val = 100000
            for action in non_quiescent_actions:
                current_state.move(action[0], action[1], self.player_symbol)
                value = self.quiescence_search(current_state, depth + 1, True, alpha, beta)
                best_val = min(best_val, value)
                if best_val <= alpha:
                    return best_val
                beta = min(best_val, beta)
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
            # action_value = self.alpha_beta_solve(current_state, 1, False, -10000000, 10000000)
            action_value = self.minimax(current_state, 1, not maximizing_player)

            if func(action_value, best_value):
                best_value = action_value
                best_action = action
        return best_action