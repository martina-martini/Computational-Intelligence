from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import random
from game import Game, Move, Player
import pickle

class QLearningPlayer(Player):
    def __init__(self, player_symbol, epsilon, epsilon_decay, epsilon_min, alpha, gamma, replay_buffer_size=10000):
        super().__init__()
        self.player_symbol = player_symbol
        self.q_values = {}
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=replay_buffer_size)

    def do_it_random(self, game):
        state_key = tuple(map(tuple, game.get_board()))
        available_actions = self.get_available_actions(game.get_board())
        action = random.choice(available_actions)
        self.q_values[(state_key, action)] = self.epsilon
        return action 

    def get_q_value(self, state, action):
        state_key = tuple(map(tuple, state))
        return self.q_values.get((state_key, action), 0.0)

    def make_move(self, game):
        state_key = tuple(map(tuple, game.get_board()))
        available_actions = self.get_available_actions(game.get_board())
        
        if random.random() <= self.epsilon:
            best_move = self.do_it_random(game)
        else:
            q_values = [self.get_q_value(state_key, action) for action in available_actions]
            max_q_value = max(q_values)
            best_moves = [action for action, q_value in zip(available_actions, q_values) if q_value == max_q_value]
            
            if not best_moves:
                best_move = random.choice(available_actions)
            else:
                best_move = random.choice(best_moves)
        
        return best_move

    def update_q_value(self, state, action, reward, next_state):
        state_key = tuple(map(tuple, state))
        next_state_key = tuple(map(tuple, next_state))
        next_max_q = max([self.get_q_value(next_state_key, next_action) for next_action in self.get_available_actions(next_state)], default=0.0)
        self.q_values[(state_key, action)] = (1 - self.alpha) * self.get_q_value(state_key, action) + self.alpha * (reward + self.gamma * next_max_q)

    def get_available_actions(self, board):
        actions = []
        size = board.shape[0]
        for x in range(size):
            for y in range(size):
                if x == 0 or x == size - 1 or y == 0 or y == size - 1:
                    if board[x, y] == -1 or board[x, y] == self.player_symbol:
                        if x == 0:
                            actions.append(((x, y), Move.BOTTOM))
                        if x == size - 1:
                            actions.append(((x, y), Move.TOP))
                        if y == 0:
                            actions.append(((x, y), Move.RIGHT))
                        if y == size - 1:
                            actions.append(((x, y), Move.LEFT))
        return actions

    def feed_reward(self, state, reward, action):
        state_key = tuple(map(tuple, state))
        if (state_key, action) in self.q_values:
            self.q_values[(state_key, action)] += self.alpha * (reward + self.gamma * self.q_values[(state_key, action)] - self.q_values[(state_key, action)])
            r = self.q_values[(state_key, action)]
        else:
            self.q_values[(state_key, action)] = -self.epsilon  
            r = -self.epsilon
        return r

    def save_q_table(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.q_values, file)

    def load_q_table(self, file_path):
        with open(file_path, 'rb') as file:
            self.q_values = pickle.load(file)

    def store_experience(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def replay_experiences(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state in batch:
            self.update_q_value(state, action, reward, next_state)