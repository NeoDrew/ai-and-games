from copy import deepcopy

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from . import HexHexAgent
from . import MCTSAgent


class HexHexMCTSAgent(AgentBase):

    _choices: list[Move]
    _board_size: int = 11
    _saturation_threshold: float = 0.5

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]
        self.hexhexAgent = HexHexAgent.HexHexAgent(self._colour)
        # self.MCTSAgent = MCTSAgent.MCTSAgent(self._colour)
        self.is_board_saturated = False
        self.current_board_state = None

    def _is_board_saturated(self, board: Board) -> bool:
        """Sets the board as saturated if more than n% of tiles are occupied."""
        if self.is_board_saturated:
            return True
        occupied_tiles = sum(
            1 for i in range(board.size) for j in range(board.size)
            if board.tiles[i][j].colour != None
        )
        total_tiles = board.size * board.size
        if (occupied_tiles / total_tiles >= self._saturation_threshold):
            self.is_board_saturated = True
            return True
        return False

    def validate_board_state(self, board: Board) -> None:
        """Ensure board state is consistent"""
        self.current_board_state = deepcopy(board)

    def is_valid_move(self, board: Board, move: tuple[int, int]) -> bool:
        """Enhanced move validation"""
        if not (0 <= move[0] < board.size and 0 <= move[1] < board.size):
            return False
        # Check both current state and provided board
        if self.current_board_state:
            if self.current_board_state.tiles[move[0]][move[1]].colour is not None:
                return False
        return board.tiles[move[0]][move[1]].colour is None

    def _check_move_for_immediate_win(self, board: Board, move: tuple[int, int]) -> bool:
        """Check if a move leads to immediate win"""
        test_board = deepcopy(board)
        if not self.is_valid_move(test_board, move):
            return False
        test_board.set_tile_colour(move[0], move[1], self.colour)
        test_board.has_ended(self.colour)
        return test_board._winner == self.colour

    def _check_all_moves_for_immediate_win(self, board: Board) -> Move | None:
        """Check all possible moves for immediate win"""
        for move in self._choices:
            if self._check_move_for_immediate_win(board, move):
                return Move(move[0], move[1])
        return None

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:

        # self.validate_board_state(board)

        if turn > self._board_size:
            winning_move = self._check_all_moves_for_immediate_win(board)
            if winning_move:
                return winning_move

        # if self._is_board_saturated(board):
        #     return self.MCTSAgent.make_move(turn, board, opp_move)

        return self.hexhexAgent.make_move(turn, board, opp_move)
