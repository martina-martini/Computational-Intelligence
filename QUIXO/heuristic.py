import random
from game import Game, Move, Player


def heuristic(game: Game) -> int:

    board = game.get_board()

    score = 0

    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            if board[x, y]!= -1:  # for each piece on the board
                if board[x, y] == game.get_current_player():
                    score += 1
                else:
                    score -= 1

    for action in game.get_non_quiescent_actions():
        from_pos, slide = action
        to_pos = (from_pos[0], from_pos[1] - 1) if slide == Move.LEFT else (from_pos[0], from_pos[1] + 1) if slide == Move.RIGHT else (from_pos[0] - 1, from_pos[1]) if slide == Move.TOP else (from_pos[0] + 1, from_pos[1])
        if 0 <= to_pos[0] < board.shape[0] and 0 <= to_pos[1] < board.shape[1] and board[to_pos] == -1:
            score += 1
        elif 0 <= to_pos[0] < board.shape[0] and 0 <= to_pos[1] < board.shape[1] and board[to_pos] == ((game.get_current_player() + 1) % 2):
            score -= 1
        # The score is increased by 1 for each potential winning line that the current player can create, and decreased by 1 for each potential winning line that the opponent can create.
    return score   
