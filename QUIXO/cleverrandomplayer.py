import random
from game import Game, Player, Move

class CleverRandomPlayer(Player):
    def __init__(self,player_symbol):
        super().__init__()
        self.player_symbol=player_symbol

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        valid_move = False
        while not valid_move:
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            valid_move = game.move(from_pos, move, self.player_symbol)
        return from_pos, move 
     