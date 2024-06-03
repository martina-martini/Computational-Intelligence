import random
from game import Game, Player, Move

class RandomPlayer(Player):
    def __init__(self,player_symbol):
        super().__init__()
        self.player_symbol=player_symbol
        
    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move  
