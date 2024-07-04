"""
TODO: Implement improved version of MCTS player for Gomoku.
* You could try different tree policy (in select), rollout policy (in simulate), or other improvements.
"""

import numpy as np
import math

from ..player import Player
from ..game import Gomoku
from typing import List, Tuple
import random

WIN = 1
LOSE = -1
DRAW = 0
NUM_SIMULATIONS = 1000
DEPTH = 2
SEED = 2024
random.seed(SEED)


class TreeNode:
    def __init__(
        self, game_state: Gomoku, player_letter: str, parent=None, parent_action=None
    ):
        self.player = player_letter
        self.game_state = game_state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.N = 0
        self.Q = 0
        self.depth = DEPTH
        self.memo = {}

    def select(self) -> "TreeNode":
        """
        Select the best child node based on UCB1 formula. Keep selecting until a leaf node is reached.
        """
        ######### YOUR CODE HERE #########
        current_node = self
        while not current_node.is_leaf_node():
            current_node = current_node.best_child()
        return current_node
        ######### YOUR CODE HERE #########

    def expand(self) -> "TreeNode":
        """
        Expand the current node by adding all possible child nodes. Return one of the child nodes for simulation.
        """
        ######### YOUR CODE HERE #########
        available_moves = self.game_state.empty_cells()
        for move in available_moves:
            new_game_state = self.game_state.copy()
            new_game_state.set_move(move[0], move[1], self.player)
            new_player = "O" if self.player == "X" else "X"
            child_node = TreeNode(
                new_game_state, new_player, parent=self, parent_action=move
            )
            self.children.append(child_node)
        if not self.children:
            return self
        return random.choice(self.children)
        ######### YOUR CODE HERE #########

    def simulate(self, depth_limit=69) -> int:
        """
        Run simulation from the current node.
        Use MCTS for shallow depths, switch to Minimax for deeper depths.
        """
        current_simulation_game = self.game_state.copy()
        current_player = self.player
        current_depth = 0

        while not current_simulation_game.game_over():
            available_moves = current_simulation_game.empty_cells()
            if not available_moves or current_depth >= depth_limit:
                break

            if current_depth < depth_limit // 2:
                # Use MCTS-style random rollout for shallow depths
                move = random.choice(available_moves)
            else:
                # Switch to Minimax evaluation for deeper depths
                move = self.alphabeta_get_move(current_simulation_game, current_player)

            current_simulation_game.set_move(move[0], move[1], current_player)
            current_player = "O" if current_player == "X" else "X"
            current_depth += 1

        if current_simulation_game.wins(self.player):
            return WIN
        elif current_simulation_game.wins("O" if self.player == "X" else "X"):
            return LOSE
        else:
            return DRAW
        ######### YOUR CODE HERE #########

    def alphabeta_get_move(self, game_state, player_letter) -> Tuple[int, int]:
        alpha = -math.inf
        beta = math.inf
        current_depth = self.depth
        move, _ = self.minimax(
            game_state,
            current_depth,
            player_letter,
            alpha,
            beta,
        )
        return move

    def minimax(self, game, depth, player_letter, alpha, beta):
        opponent_letter = "O" if player_letter == "X" else "X"

        if depth == 0 or game.game_over():
            return [-1, -1], self.evaluate(game)

        best_move = [-1, -1]

        if player_letter == self.player:
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
                _, score = self.minimax(game, depth - 1, self.player, alpha, beta)
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

        opponent_letter = "O" if self.player == "X" else "X"

        player_score = evaluate_patterns(state, self.player)
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

    def backpropagate(self, result: int):
        """
        Backpropagate the result of the simulation to the root node.
        """
        ######### YOUR CODE HERE #########
        if self.parent:
            self.parent.backpropagate(-result)
        self.N += 1
        self.Q += result
        ######### YOUR CODE HERE #########

    def is_leaf_node(self) -> bool:
        return len(self.children) == 0

    def is_terminal_node(self) -> bool:
        return self.game_state.game_over()

    def best_child(self) -> "TreeNode":
        return max(self.children, key=lambda c: c.ucb())

    def ucb(self, c=math.sqrt(2)) -> float:
        return self.Q / (1 + self.N) + c * np.sqrt(np.log(self.parent.N) / (1 + self.N))


class GMK_BetterMCTS(Player):
    def __init__(self, letter, num_simulations=NUM_SIMULATIONS):
        super().__init__(letter)
        self.num_simulations = num_simulations

    def get_move(self, game: Gomoku):
        mtcs = TreeNode(game, self.letter)
        for num in range(self.num_simulations):
            leaf = mtcs.select()
            if not leaf.is_terminal_node():
                leaf.expand()
            result = leaf.simulate()
            leaf.backpropagate(-result)

        best_child = max(mtcs.children, key=lambda c: c.N)
        return best_child.parent_action

    def __str__(self) -> str:
        return "Better MCTS Player"
