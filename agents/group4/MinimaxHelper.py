from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile
from collections import deque
from copy import deepcopy

class MinimaxHelper:
    """A class of helper functions for MiniMax with Alpha-Beta Pruning"""
    
    def getLegalMoves(self, board: Board) -> list:
        """
            Gets a list of all legal moves based on the current board state
        
            :param board: Current state of the game board.
            :return legal_moves: A list of Move objects, denoting the available legal moves.
        """
        legal_moves = []
        tiles = board.tiles()

        for i in range(board.size()):
            for j in range(board.size()):
                if not tiles[i][j].colour():
                    #Tile is unoccupied, could move here
                    legal_moves.append(Move(i,j))

        return legal_moves

    def _getShortestPath(self, board: Board, colour: Colour) -> int:
        """
            Uses breadth-first search to find the shortest winning
            connection between two sides of the board.

            :param board: Current state of the game board.
            :param colour:  Colour of the player to find the shortest path for.
            
        """
        size = board.size()
        tiles = board.tiles()
        visited = set()
        parent = {}
        queue = deque()

        if colour == Colour.RED:
            #Top row sources
            for y in range(size):
                if tiles[0][y].colour == Colour.RED:
                    queue.append((0,y))
                    visited.add((0,y))
                    parent[(0,y)] = None
        elif colour == Colour.BLUE:
            #Left column sources
            for x in range(size):
                if tiles[x][0].colour == Colour.BLUE:
                    queue.append((x,0))
                    visited.add((x,0))
                    parent[(x,0)] = None
        else:
            return

        target = None

        while queue:
            x, y = queue.popleft()

            #Goal test
            if colour == Colour.RED and x == size - 1:
                target = (x, y)
                break
            if colour == Colour.BLUE and y == size - 1:
                target = (x, y)
                break

            #Explore neighbours
            for idx in range(Tile.NEIGHBOUR_COUNT):
                x_n = x + Tile.I_DISPLACEMENTS[idx]
                y_n = y + Tile.J_DISPLACEMENTS[idx]
                if 0 <= x_n < size and 0 <= y_n < size:
                    if (x_n, y_n) not in visited and tiles[x_n][y_n].colour == colour:
                        visited.add((x_n, y_n))
                        parent[(x_n, y_n)] = (x, y)
                        queue.append((x_n, y_n))

        #Reconstruct path if the target is reached, calculating
        #a score for the path
        if target is not None:
            score = 0
            path = []
            cur = target
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
                score += 1
            return score
        else:
            #There is no winning path available
            return float('inf')

    def getMoveScore(self, board: Board, move: Move, colour: Colour) -> int:
        """
            Calculates a heuristic score of a given move

            :param board: Current state of the game board.
            :param move: The move to assign a value for.
            :param colour: The colour of the player to compute the move value for.
            :return: The score of the given move
        """
        board_cpy = deepcopy(board)
        tiles = board_cpy.tiles()
        if tiles[move.x()][move.y()].colour():
            #Tile already occupied
            return float('-inf')
        board_cpy.set_tile_colour(move.x(), move.y(), colour)

        best_for_me = self._getShortestPath(board_cpy, colour)
        best_for_them = self._getShortestPath(board_cpy, colour.opposite())

        if best_for_me == float('inf') and best_for_them == float('inf'):
            #Neither side can win, treat this move as "neutral"
            return 0

        return best_for_them - best_for_me

    def orderMoves(self, board: Board, moves: list, colour: Colour) -> list:
        """
            Returns moves ordered by their heuristic score in descending order

            :param board: Current state of the game board.
            :param moves: List of Moves to assign scores to.
            :param colour: Colour of the current player.
            :return: A list of sorted Move objects
        """
        return sorted(moves, key=lambda m: self.getMoveScore(board, m, colour), reverse=True)