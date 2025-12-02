from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile
from collections import deque
from copy import deepcopy
import heapq

class MinimaxHelper:
    """A class of helper functions for MiniMax with Alpha-Beta Pruning"""
    
    @staticmethod
    def getLegalMoves(board: Board) -> list:
        """
            Gets a list of all legal moves based on the current board state
        
            :param board: Current state of the game board.
            :return legal_moves: A list of Move objects, denoting the available legal moves.
        """
        legal_moves = []
        tiles = board.tiles

        for i in range(board.size):
            for j in range(board.size):
                if not tiles[i][j].colour:
                    #Tile is unoccupied, could move here
                    legal_moves.append(Move(i,j))

        return legal_moves

    @staticmethod
    def _getShortestPath(board: Board, colour: Colour) -> int:
        """
            Uses Dijkstra to find the shortest winning
            connection between two sides of the board.

            :param board: Current state of the game board.
            :param colour:  Colour of the player to find the shortest path for.
            
        """
        size = board.size
        tiles = board.tiles

        def cost(x, y):
            """Computes the cost of placing a move at this tile"""
            tile = tiles[x][y].colour
            if tile == colour:
                return 0
            elif tile == None:
                return 1
            else:
                return float('inf')

        pq = []
        visited = set()

        if colour == Colour.RED:
            #Top row sources
            for y in range(size):
                heapq.heappush(pq, (cost(0, y), 0, y))
        else:
            #Left row sources
            for x in range(size):
                heapq.heappush(pq, (cost(x, 0), x, 0))

        while pq:
            dist, x, y = heapq.heappop(pq)

            if (x, y) in visited:
                continue
            visited.add((x, y))

            #Goal test
            if colour == Colour.RED and x == size - 1:
                return dist
            if colour == Colour.BLUE and y == size - 1:
                return dist

            #Add neighbours to heap
            for idx in range(Tile.NEIGHBOUR_COUNT):
                nx = x + Tile.I_DISPLACEMENTS[idx]
                ny = y + Tile.J_DISPLACEMENTS[idx]
                if 0 <= nx < size and 0 <= ny < size:
                    heapq.heappush(pq, (dist + cost(nx, ny), nx, ny))

        return float('inf')

    @staticmethod
    def getMoveScore(board: Board, move: Move, colour: Colour) -> int:
        """
            Calculates a heuristic score of a given move

            :param board: Current state of the game board.
            :param move: The move to assign a value for.
            :param colour: The colour of the player to compute the move value for.
            :return: The score of the given move
        """
        board_cpy = deepcopy(board)
        tiles = board_cpy.tiles
        if tiles[move.x][move.y].colour:
            #Tile already occupied
            return float('-inf')
        board_cpy.set_tile_colour(move.x, move.y, colour)

        best_for_me = MinimaxHelper._getShortestPath(board_cpy, colour)
        best_for_them = MinimaxHelper._getShortestPath(board_cpy, Colour.opposite(colour))

        if best_for_me == float('inf') and best_for_them == float('inf'):
            #Neither side can win, treat this move as "neutral"
            return 0

        return best_for_them - best_for_me

    @staticmethod
    def getOrderedMoves(board: Board, colour: Colour) -> list:
        """
            Returns moves ordered by their heuristic score in descending order

            :param board: Current state of the game board.
            :param moves: List of Moves to assign scores to.
            :param colour: Colour of the current player.
            :return: A list of sorted Move objects
        """
        moves = MinimaxHelper.getLegalMoves(board)
        return sorted(moves, key=lambda m: MinimaxHelper.getMoveScore(board, m, colour), reverse=True)