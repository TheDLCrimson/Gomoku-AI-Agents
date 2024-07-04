"""
TODO: Implement Minimax with Alpha-Beta Pruning player for Gomoku.
* You need to implement heuristic evaluation function for non-terminal states.
* Optional: You can implement the function promising_next_moves to explore reduce the branching factor.
"""

from ..player import Player
from ..game import Gomoku
from typing import List, Tuple
import math
import random

SEED = 2024
random.seed(SEED)

DEPTH = 2  # Define the depth of the search tree.


class GMK_AlphaBetaPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
        self.depth = DEPTH
        self.memo = {}

    def get_move(self, game: Gomoku):
        if game.last_move == (-1, -1):
            mid_size = game.size // 2
            moves = [
                (mid_size, mid_size),
                (mid_size - 1, mid_size - 1),
                (mid_size + 1, mid_size + 1),
                (mid_size - 1, mid_size + 1),
                (mid_size + 1, mid_size - 1),
            ]
            move = random.choice(moves)
            while not game.valid_move(move[0], move[1]):
                move = random.choice(moves)
            return move
        else:
            alpha = -math.inf
            beta = math.inf
            current_depth = self.depth
            move, _ = self.minimax(game, current_depth, self.letter, alpha, beta)
        return move

    def minimax(self, game, depth, player_letter, alpha, beta):
        """
        AI function that chooses the best move with alpha-beta pruning.
        :param game: current state of the board
        :param depth: node index in the tree (0 <= depth <= 9)
        :param player_letter: value representing the player
        :param alpha: best value that the maximizer can guarantee
        :param beta: best value that the minimizer can guarantee
        :return: a list with [best row, best col] and the best score
        """

        opponent_letter = "O" if player_letter == "X" else "X"

        if depth == 0 or game.game_over():
            return [-1, -1], self.evaluate(game)

        best_move = [-1, -1]

        if player_letter == self.letter:
            best_score = -math.inf
            for x, y in self.promising_next_moves(game, player_letter):
                game.set_move(x, y, player_letter)
                _, score = self.minimax(game, depth - 1, opponent_letter, alpha, beta)
                game.reset_move(x, y)
                if score > best_score:
                    best_score = score
                    best_move = [x, y]
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
        else:
            best_score = math.inf
            for x, y in self.promising_next_moves(game, player_letter):
                game.set_move(x, y, player_letter)
                _, score = self.minimax(game, depth - 1, self.letter, alpha, beta)
                game.reset_move(x, y)
                if score < best_score:
                    best_score = score
                    best_move = [x, y]
                beta = min(beta, score)
                if alpha >= beta:
                    break

        return best_move, best_score

    def evaluate(self, game, state=None) -> float:
        if state is None:
            state = game.board_state

        hashed_state = tuple(tuple(row) for row in state)
        if hashed_state in self.memo:
            return self.memo[hashed_state]

        def score_line(line, player_letter):
            opponent_letter = "O" if player_letter == "X" else "X"
            player_count = line.count(player_letter)
            opponent_count = line.count(opponent_letter)
            empty_count = line.count(None)

            if opponent_count == 0:
                return patterns.get((player_count, empty_count), 0)
            elif player_count == 0:
                return -opponent_patterns.get((opponent_count, empty_count), 0)
            return 0

        def evaluate_patterns(state, player_letter):
            total_score = 0

            # Check rows
            for x in range(game.size):
                for y in range(game.size - 4):
                    line = [state[x][y + i] for i in range(5)]
                    total_score += score_line(line, player_letter)

            # Check columns
            for y in range(game.size):
                for x in range(game.size - 4):
                    line = [state[x + i][y] for i in range(5)]
                    total_score += score_line(line, player_letter)

            # Check diagonal (\)
            for x in range(game.size - 4):
                for y in range(game.size - 4):
                    line = [state[x + i][y + i] for i in range(5)]
                    total_score += score_line(line, player_letter)

            # Check diagonal (/)
            for x in range(game.size - 4):
                for y in range(4, game.size):
                    line = [state[x + i][y - i] for i in range(5)]
                    total_score += score_line(line, player_letter)

            return total_score

        # Patterns for player and opponent with different weights
        patterns = {
            (5, 0): 1000000,  # Five in a row
            (4, 1): 100000,  # Four in a row with one empty space
            (4, 0): 50000,  # Four in a row blocked at one end
            (3, 2): 50000,  # Three in a row with two empty spaces
            (3, 1): 30000,  # Three in a row blocked at one end
            (2, 2): 10000,  # Two in a row with two empty spaces
            (2, 1): 5000,  # Two in a row blocked at one end
            (1, 2): 1000,  # Single stone with two empty spaces
            (1, 1): 500,  # Single stone blocked at one end
        }
        opponent_patterns = {
            (5, 0): 1000000,  # Five in a row (opponent)
            (4, 1): 80000,  # Four in a row with one empty space (opponent)
            (4, 0): 40000,  # Four in a row blocked at one end (opponent)
            (3, 2): 30000,  # Three in a row with two empty spaces (opponent)
            (3, 1): 15000,  # Three in a row blocked at one end (opponent)
            (2, 2): 7000,  # Two in a row with two empty spaces (opponent)
            (2, 1): 3000,  # Two in a row blocked at one end (opponent)
            (1, 2): 500,  # Single stone with two empty spaces (opponent)
            (1, 1): 250,  # Single stone blocked at one end (opponent)
        }

        opponent_letter = "O" if self.letter == "X" else "X"

        player_score = evaluate_patterns(state, self.letter)
        opponent_score = evaluate_patterns(state, opponent_letter)

        score = player_score - 1.05 * opponent_score

        self.memo[hashed_state] = score
        return score

    def promising_next_moves(self, game, player_letter) -> List[Tuple[int]]:
        moves = game.empty_cells()
        if not moves:
            return moves

        def move_score(move):
            x, y = move
            score = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < game.size and 0 <= ny < game.size:
                        if game.board_state[nx][ny] == player_letter:
                            score += 2
                        elif game.board_state[nx][ny] != None:
                            score -= 1
            return score

        scored_moves = [(move_score(move), move) for move in moves]
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        sorted_moves = [move for _, move in scored_moves]

        return sorted_moves

    def __str__(self):
        return "AlphaBeta Player"
