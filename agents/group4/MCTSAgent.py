from __future__ import annotations

from math import sqrt, log
from random import Random
from typing import Optional, Tuple, List, Dict

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile


Coord = Tuple[int, int]


class RaveMCTSNode:
    """
    Fast RAVE node:
      - does not store a board
      - stores move that led here
      - stores player-to-move at this node
      - stores AMAF (RAVE) stats at THIS node keyed by move index 0..(size*size-1)
    """

    __slots__ = (
        "parent",
        "move",
        "to_play",
        "children",
        "wins",
        "visits",
        "untried_moves",
        "amaf_wins",
        "amaf_visits",
    )

    def __init__(
        self,
        parent: Optional["RaveMCTSNode"],
        move: Optional[Coord],
        to_play: Colour,
        board_area: int,
    ):
        self.parent = parent
        self.move = move
        self.to_play = to_play

        self.children: List[RaveMCTSNode] = []
        self.wins = 0.0
        self.visits = 0

        # list[Coord] generated lazily, using current search state
        self.untried_moves: Optional[List[Coord]] = None

        # AMAF arrays, allocated lazily to save memory
        self.amaf_wins: Optional[List[float]] = None
        self.amaf_visits: Optional[List[int]] = None

    @staticmethod
    def _beta(rave_const: float, child_visits: int) -> float:
        return sqrt(rave_const / (3.0 * child_visits + rave_const))

    def _ensure_amaf(self, board_area: int) -> None:
        if self.amaf_wins is None:
            self.amaf_wins = [0.0] * board_area
            self.amaf_visits = [0] * board_area

    def amaf_q(self, move_idx: int) -> float:
        if self.amaf_visits is None:
            return 0.0
        v = self.amaf_visits[move_idx]
        if v == 0:
            return 0.0
        return self.amaf_wins[move_idx] / v

    def rave_score_child(
        self,
        child: "RaveMCTSNode",
        explore: float,
        rave_const: float,
        parent_visits: int,
        child_move_idx: int,
    ) -> float:
        # unvisited children should be explored immediately
        if child.visits == 0:
            return float("inf")

        # UCT component (root player win rate)
        q = child.wins / child.visits
        u = explore * sqrt(log(max(1, parent_visits)) / child.visits)
        mcts_value = q + u

        # AMAF component uses parent's AMAF stats for this move
        amaf_value = self.amaf_q(child_move_idx)

        beta = self._beta(rave_const, child.visits)
        return (1.0 - beta) * mcts_value + beta * amaf_value


class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)

        self.colour = colour
        self.opposing_colour = Colour.opposite(self.colour)

        # Tuning
        self.simulations = 1000
        self.explore = 1.41
        self.rave_constant = 300.0
        self.win_score = 1.0

        # Rollout tuning, speed oriented
        self.rollout_neighbor_bias = 0.70
        self.rollout_check_min_plies = 6
        self.rollout_check_every = 3

        # Early stop
        self.early_stop_threshold = 0.95
        self.min_visits_ratio = 0.12
        self.min_sims_before_early_stop = 120

        # Heuristic ordering (used only for expansion, not for full rollouts)
        self.friendly_neighbor_weight = 0.5
        self.bridge_weight = 0.3
        self.edge_control_weight = 0.4

        # RNG
        self._rng = Random(42)

        # Fixed board config (your environment uses 11)
        self._board_size = 11
        self._board_area = self._board_size * self._board_size
        self._all_positions = [(i, j) for i in range(
            self._board_size) for j in range(self._board_size)]

        # Zobrist for optional caching if you extend it later
        self._zobrist_table = {
            (i, j, color): self._rng.getrandbits(64)
            for i in range(self._board_size)
            for j in range(self._board_size)
            for color in [Colour.RED, Colour.BLUE]
        }

        self.root: Optional[RaveMCTSNode] = None

    def _move_idx(self, move: Coord) -> int:
        return move[0] * self._board_size + move[1]

    def _neighbors(self, x: int, y: int) -> List[Coord]:
        out: List[Coord] = []
        for k in range(Tile.NEIGHBOUR_COUNT):
            nx = x + Tile.I_DISPLACEMENTS[k]
            ny = y + Tile.J_DISPLACEMENTS[k]
            if 0 <= nx < self._board_size and 0 <= ny < self._board_size:
                out.append((nx, ny))
        return out

    def _compute_initial_empties(self, board: Board) -> Tuple[List[Coord], Dict[Coord, int], List[Coord]]:
        empties: List[Coord] = []
        pos_to_idx: Dict[Coord, int] = {}
        occupied: List[Coord] = []

        for x in range(board.size):
            row = board.tiles[x]
            for y in range(board.size):
                if row[y].colour is None:
                    pos_to_idx[(x, y)] = len(empties)
                    empties.append((x, y))
                else:
                    occupied.append((x, y))

        return empties, pos_to_idx, occupied

    def _remove_empty(self, empties: List[Coord], pos_to_idx: Dict[Coord, int], move: Coord):
        """
        Remove move from empties in O(1) using swap-with-last.
        Returns an undo record.
        """
        idx = pos_to_idx.get(move)
        if idx is None:
            return None

        last = empties[-1]
        if last == move:
            empties.pop()
            del pos_to_idx[move]
            return ("noswap", move)

        empties[idx] = last
        pos_to_idx[last] = idx
        empties.pop()
        del pos_to_idx[move]
        return ("swap", move, idx, last)

    def _undo_remove_empty(self, empties: List[Coord], pos_to_idx: Dict[Coord, int], rec):
        if rec is None:
            return
        tag = rec[0]
        if tag == "noswap":
            move = rec[1]
            pos_to_idx[move] = len(empties)
            empties.append(move)
            return

        _, move, idx, last = rec
        # currently, "last" is at position idx
        empties.append(last)
        empties[idx] = move
        pos_to_idx[move] = idx
        pos_to_idx[last] = len(empties) - 1

    def _evaluate_move_for_ordering(self, board: Board, move: Coord) -> float:
        x, y = move
        score = 0.0
        center = board.size // 2

        dist_to_center = abs(x - center) + abs(y - center)
        score += max(0.0, (board.size - dist_to_center)) / board.size

        friendly_neighbors = 0
        bridge_score = 0
        for nx, ny in self._neighbors(x, y):
            c = board.tiles[nx][ny].colour
            if c == self.colour:
                friendly_neighbors += 1
                if abs(nx - x) == 1 and abs(ny - y) == 1:
                    bridge_score += 1

        score += friendly_neighbors * self.friendly_neighbor_weight
        score += bridge_score * self.bridge_weight

        if self.colour == Colour.RED and (x == 0 or x == board.size - 1):
            score += self.edge_control_weight
        elif self.colour == Colour.BLUE and (y == 0 or y == board.size - 1):
            score += self.edge_control_weight

        return score

    def _generate_candidate_moves(
        self,
        board: Board,
        empties: List[Coord],
        pos_to_idx: Dict[Coord, int],
        occupied_base: List[Coord],
        path_moves: List[Coord],
        limit: int = 40,
    ) -> List[Coord]:
        """
        Fast candidate generation for expansion:
          - take empty neighbors of already occupied cells (initial + path)
          - fallback to a few random empties
          - then sort candidates by heuristic ordering
        """
        cand_set = set()

        # if early game, play center first if possible
        center = board.size // 2
        if (center, center) in pos_to_idx:
            cand_set.add((center, center))

        # neighbors of occupied cells
        # using (occupied_base + last few path moves) keeps this cheap
        recent = path_moves[-12:] if len(path_moves) > 12 else path_moves
        for x, y in occupied_base:
            for nx, ny in self._neighbors(x, y):
                if (nx, ny) in pos_to_idx:
                    cand_set.add((nx, ny))
                    if len(cand_set) >= limit:
                        break
            if len(cand_set) >= limit:
                break

        if len(cand_set) < limit:
            for x, y in recent:
                for nx, ny in self._neighbors(x, y):
                    if (nx, ny) in pos_to_idx:
                        cand_set.add((nx, ny))
                        if len(cand_set) >= limit:
                            break
                if len(cand_set) >= limit:
                    break

        # fallback: sample a few random empties
        if len(cand_set) < 10 and empties:
            sample_count = min(10, len(empties))
            for _ in range(sample_count):
                cand_set.add(empties[self._rng.randrange(len(empties))])

        cands = list(cand_set)
        cands.sort(key=lambda m: self._evaluate_move_for_ordering(
            board, m), reverse=True)
        return cands

    def _select(self, root: RaveMCTSNode, board: Board, empties: List[Coord], pos_to_idx: Dict[Coord, int], occupied_base: List[Coord]):
        """
        Selection, applies moves to board and updates empties.
        Returns: (node, path_stack, path_moves)
        path_stack contains undo info for empties and board colour.
        """
        node = root
        path_stack = []  # list of (move, prev_colour, empties_undo_record)
        path_moves: List[Coord] = []

        while True:
            # not expanded yet
            if node.untried_moves is None:
                return node, path_stack, path_moves

            # still has untried moves
            if node.untried_moves:
                return node, path_stack, path_moves

            if not node.children:
                return node, path_stack, path_moves

            # choose child by RAVE score, minimax style by side to play
            best_children: List[RaveMCTSNode] = []
            if node.to_play == self.colour:
                best_score = float("-inf")
                for c in node.children:
                    mi = self._move_idx(c.move)
                    s = node.rave_score_child(
                        c, self.explore, self.rave_constant, node.visits, mi)
                    if s > best_score:
                        best_score = s
                        best_children = [c]
                    elif s == best_score:
                        best_children.append(c)
            else:
                best_score = float("inf")
                for c in node.children:
                    mi = self._move_idx(c.move)
                    s = node.rave_score_child(
                        c, self.explore, self.rave_constant, node.visits, mi)
                    if s < best_score:
                        best_score = s
                        best_children = [c]
                    elif s == best_score:
                        best_children.append(c)

            child = best_children[self._rng.randrange(len(best_children))]

            # apply child move by current node.to_play
            move = child.move
            x, y = move
            prev_colour = board.tiles[x][y].colour
            board.set_tile_colour(x, y, node.to_play)
            empties_undo = self._remove_empty(empties, pos_to_idx, move)

            path_stack.append((move, prev_colour, empties_undo, node.to_play))
            path_moves.append(move)

            node = child

    def _expand(self, node: RaveMCTSNode, board: Board, empties: List[Coord], pos_to_idx: Dict[Coord, int], occupied_base: List[Coord], path_moves: List[Coord]):
        """
        Expand one move from node if possible, applies it and returns (new_node, applied_move_stack_record_or_None).
        """
        if node.untried_moves is None:
            node.untried_moves = self._generate_candidate_moves(
                board, empties, pos_to_idx, occupied_base, path_moves)

        # remove any candidates that are no longer empty
        while node.untried_moves and node.untried_moves[0] not in pos_to_idx:
            node.untried_moves.pop(0)

        if not node.untried_moves:
            return node, None

        move = node.untried_moves.pop(0)

        child = RaveMCTSNode(
            parent=node,
            move=move,
            to_play=Colour.opposite(node.to_play),
            board_area=self._board_area,
        )
        node.children.append(child)

        # apply move by node.to_play
        x, y = move
        prev_colour = board.tiles[x][y].colour
        board.set_tile_colour(x, y, node.to_play)
        empties_undo = self._remove_empty(empties, pos_to_idx, move)

        applied = (move, prev_colour, empties_undo, node.to_play)
        return child, applied

    def _rollout(self, board: Board, to_play: Colour, empties: List[Coord], pos_to_idx: Dict[Coord, int]):
        """
        Cheap rollout:
          - mostly picks a random empty neighbor of last move if available
          - otherwise random empty
          - checks win occasionally
        Returns:
          (result_root_wins, rollout_played_moves as list[(move, player)])
        """
        rollout_stack = [
        ]  # undo info: (move, prev_colour, empties_undo, player)
        played: List[Tuple[Coord, Colour]] = []

        current = to_play
        moves_made = 0
        last_move: Optional[Coord] = None

        while empties:
            move: Optional[Coord] = None

            # neighbor bias, but only if last_move exists
            if last_move is not None and self._rng.random() < self.rollout_neighbor_bias:
                lx, ly = last_move
                neighs = self._neighbors(lx, ly)
                self._rng.shuffle(neighs)
                for nm in neighs:
                    if nm in pos_to_idx:
                        move = nm
                        break

            if move is None:
                move = empties[self._rng.randrange(len(empties))]

            x, y = move
            prev_colour = board.tiles[x][y].colour
            board.set_tile_colour(x, y, current)
            empties_undo = self._remove_empty(empties, pos_to_idx, move)

            rollout_stack.append((move, prev_colour, empties_undo, current))
            played.append((move, current))

            moves_made += 1
            last_move = move

            # terminal check throttled
            if moves_made >= self.rollout_check_min_plies and (moves_made % self.rollout_check_every == 0):
                if board.has_ended(current):
                    result = (board._winner == self.colour)
                    # undo rollout
                    for mv, prevc, urec, _pl in reversed(rollout_stack):
                        board.set_tile_colour(mv[0], mv[1], prevc)
                        self._undo_remove_empty(empties, pos_to_idx, urec)
                    return result, played

            current = Colour.opposite(current)

        # if empties exhausted, do a final check for current player who just moved
        # played is non-empty if board filled
        if played:
            last_player = played[-1][1]
            board.has_ended(last_player)

        result = (board._winner == self.colour)

        for mv, prevc, urec, _pl in reversed(rollout_stack):
            board.set_tile_colour(mv[0], mv[1], prevc)
            self._undo_remove_empty(empties, pos_to_idx, urec)

        return result, played

    def _backprop(self, node: RaveMCTSNode, result_root_wins: bool, rollout_moves: List[Tuple[Coord, Colour]]):
        """
        Backprop:
          - standard: wins/visits are from root player perspective
          - AMAF: for each node, update AMAF for moves played by node.to_play
        """
        while node is not None:
            node.visits += 1
            if result_root_wins:
                node.wins += self.win_score

            node._ensure_amaf(self._board_area)

            # If node.to_play ultimately "wins" from node's perspective:
            # root_wins True means self.colour won.
            # node.to_play won iff node.to_play == self.colour when root_wins is True
            # otherwise node.to_play won iff node.to_play != self.colour
            node_to_play_won = (
                node.to_play == self.colour) == result_root_wins

            # update AMAF for moves made by node.to_play during rollout
            aw = node.amaf_wins
            av = node.amaf_visits
            for mv, pl in rollout_moves:
                if pl != node.to_play:
                    continue
                idx = self._move_idx(mv)
                av[idx] += 1
                if node_to_play_won:
                    aw[idx] += self.win_score

            node = node.parent

    def _undo_path(self, board: Board, empties: List[Coord], pos_to_idx: Dict[Coord, int], path_stack):
        for mv, prevc, urec, _pl in reversed(path_stack):
            board.set_tile_colour(mv[0], mv[1], prevc)
            self._undo_remove_empty(empties, pos_to_idx, urec)

    def check_immediate_win(self, board: Board, move: Coord) -> bool:
        x, y = move
        if board.tiles[x][y].colour is not None:
            return False
        board.set_tile_colour(x, y, self.colour)
        ended = board.has_ended(self.colour)
        board.set_tile_colour(x, y, None)
        return ended and board._winner == self.colour

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # Swap rule
        if turn == 2 and board.tiles[board.size // 2][board.size // 2].colour is not None:
            return Move(-1, -1)

        # Immediate win check, cheap and safe
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    if self.check_immediate_win(board, (x, y)):
                        return Move(x, y)

        # Create one working board for the entire search
        # we will apply and undo moves directly on the provided board safely per iteration
        work_board = board

        empties, pos_to_idx, occupied_base = self._compute_initial_empties(
            work_board)

        root = RaveMCTSNode(
            parent=None,
            move=None,
            to_play=self.colour,
            board_area=self._board_area,
        )
        # lazily expanded, but we can seed untried_moves immediately for speed
        root.untried_moves = self._generate_candidate_moves(
            work_board, empties, pos_to_idx, occupied_base, [])

        best_child: Optional[RaveMCTSNode] = None
        best_visits = 0

        for i in range(self.simulations):
            # Selection
            node, path_stack, path_moves = self._select(
                root, work_board, empties, pos_to_idx, occupied_base)

            # Expansion
            node2, expanded_record = self._expand(
                node, work_board, empties, pos_to_idx, occupied_base, path_moves)
            if expanded_record is not None:
                path_stack.append(expanded_record)
                path_moves.append(expanded_record[0])

            # Rollout from node2 state
            result, rollout_moves = self._rollout(
                work_board, node2.to_play, empties, pos_to_idx)

            # Backprop
            self._backprop(node2, result, rollout_moves)

            # Undo everything we played during selection and expansion
            self._undo_path(work_board, empties, pos_to_idx, path_stack)

            # Early stop
            if i >= self.min_sims_before_early_stop and root.children:
                local_best = max(root.children, key=lambda c: c.visits)
                if local_best.visits > best_visits:
                    best_visits = local_best.visits
                    best_child = local_best

                visit_ratio = best_visits / float(i + 1)
                win_rate = (best_child.wins / best_child.visits) if (
                    best_child and best_child.visits > 0) else 0.0
                if visit_ratio >= self.min_visits_ratio and win_rate >= self.early_stop_threshold:
                    break

        # Choose best move
        if root.children:
            final_best = max(root.children, key=lambda c: c.visits)
            mv = final_best.move
            if mv is not None and board.tiles[mv[0]][mv[1]].colour is None:
                self.root = final_best
                self.root.parent = None
                return Move(mv[0], mv[1])

        # Fallback random
        if empties:
            mv = empties[self._rng.randrange(len(empties))]
            self.root = None
            return Move(mv[0], mv[1])

        return Move(-1, -1)
