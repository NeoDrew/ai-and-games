from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile
from collections import deque
from group4 import MinimaxHelper
from copy import deepcopy

class MinimaxAgent(AgentBase):
    _choices: list[Move]
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """

        return self.minimaxRoot(board)

    def getScore(self, board: Board) -> int:
        """
            Calculates a heuristic score of a given move

            :param board: Current state of the game board.
            :param move: The move to assign a value for.
            :param colour: The colour of the player to compute the move value for.
            :return: The score of the given move
        """

        best_for_me = MinimaxHelper._getShortestPath(board, self.colour)
        best_for_them = MinimaxHelper._getShortestPath(board, self.colour.opposite())

        if best_for_me == float('inf') and best_for_them == float('inf'):
            #Neither side can win, treat this move as "neutral"
            return 0

        return best_for_them - best_for_me
    
    def minimaxRoot(self, board:Board, depth=3) -> Move:
        alpha=float('-inf')
        beta=float('inf')
        best_move = None
        max_eval = float('-inf')
        for move in MinimaxHelper.getOrderedMoves(board):
            board_cpy = deepcopy(board)
            board_cpy.set_tile_colour(move.x(), move.y(), self.colour)
            eval = self.minimaxVal(board_cpy, self.colour.opposite(), depth-1, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return best_move
        
    def minimaxVal(self, board: Board, current_colour: Colour, depth: int, alpha, beta) -> int:
        if depth == 0 or board.has_ended(self.colour) or board.has_ended(self.colour.opposite()):
            return self.getScore(board)
        elif current_colour == self.colour:
            max_eval = float('-inf')
            for move in MinimaxHelper.getOrderedMoves(board):
                board_cpy = deepcopy(board)
                board_cpy.set_tile_colour(move.x(), move.y(), current_colour)
                eval = self.minimaxVal(board_cpy, current_colour.opposite(), depth-1, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for move in MinimaxHelper.getOrderedMoves(board):
                board_cpy = deepcopy(board)
                board_cpy.set_tile_colour(move.x(), move.y(), current_colour)
                eval = self.minimaxVal(board_cpy, current_colour.opposite(), depth-1, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval