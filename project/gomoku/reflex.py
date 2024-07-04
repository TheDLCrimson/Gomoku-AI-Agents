"""
TODO: Implement Approximate Q-Learning player for Gomoku.
* Extract features from the state-action pair and store in a numpy array.
* Define the size of the feature vector in the feature_size method.
"""

import numpy as np
from ..player import Player


class GMK_Reflex(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        best_score = -np.inf
        best_move = None
        opponent = "O" if self.letter == "X" else "X"

        for x, y in game.empty_cells():
            game.set_move(x, y, self.letter)
            score = self.evaluate(game, self.letter, opponent)
            game.reset_move(x, y)

            if score > best_score:
                best_score = score
                best_move = (x, y)

        return best_move

    def evaluate(self, game, player, opponent):
        player_score = self.evaluate_player(game, player)
        opponent_score = self.evaluate_player(game, opponent)
        return player_score - opponent_score

    def evaluate_player(self, game, player):
        score = 0
        for x in range(game.size):
            for y in range(game.size):
                if game.board_state[x][y] == player:
                    score += self.evaluate_position(game, x, y, player)
        return score

    def evaluate_position(self, game, x, y, player):
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            score += self.evaluate_direction(game, x, y, dx, dy, player)
            score += self.evaluate_direction(game, x, y, -dx, -dy, player)

        return score

    def evaluate_direction(self, game, x, y, dx, dy, player):
        count = 1
        blocked = False
        for step in range(1, 5):
            if self.check_direction(game, x, y, dx * step, dy * step, player):
                count += 1
            elif self.check_block(game, x, y, dx * step, dy * step, player):
                blocked = True
                break
            else:
                break

        return self.get_score(count, blocked)

    def check_direction(self, game, x, y, dx, dy, player):
        nx, ny = x + dx, y + dy
        if 0 <= nx < game.size and 0 <= ny < game.size:
            return game.board_state[nx][ny] == player
        return False

    def check_block(self, game, x, y, dx, dy, player):
        nx, ny = x + dx, y + dy
        opponent = "O" if player == "X" else "X"
        if 0 <= nx < game.size and 0 <= ny < game.size:
            return game.board_state[nx][ny] == opponent
        return False

    def get_score(self, count, blocked):
        if count == 5:
            return 10000
        elif count == 4:
            return 1000 if not blocked else 500
        elif count == 3:
            return 100 if not blocked else 50
        elif count == 2:
            return 10 if not blocked else 5
        elif count == 1:
            return 1
        return 0

    def feature_size(self):
        return 10

    def update_weights(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        features = self.extract_features(state, *action, self.letter)
        q_value = np.dot(self.weights, features)

        next_best_score = -np.inf
        for x, y in state.empty_cells():
            next_features = self.extract_features(state, x, y, self.letter)
            next_score = np.dot(self.weights, next_features)
            if next_score > next_best_score:
                next_best_score = next_score

        target = reward + gamma * next_best_score
        self.weights += alpha * (target - q_value) * features

    def __str__(self):
        return "Reflex Player"
