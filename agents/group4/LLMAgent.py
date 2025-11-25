import json
import os
import google.generativeai as genai
from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Single shared model (not on self, avoids deepcopy issues)
_model = genai.GenerativeModel("gemini-1.5-pro-latest")


class LLMAgent(AgentBase):
    _choices: list[Move]
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        # Precompute all coordinates (mainly for fallback)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

    def _board_to_text(self, board: Board) -> str:
        """Convert board to text the model can read."""
        rows: list[str] = []
        for row in board._tiles:
            # tile.colour is Colour or None
            rows.append(
                " ".join(
                    tile.colour.name[0] if tile.colour is not None else "."
                    for tile in row
                )
            )
        return "\n".join(rows)

    def _legal_moves(self, board: Board) -> list[tuple[int, int]]:
        size = len(board._tiles)
        moves: list[tuple[int, int]] = []
        for x in range(size):
            for y in range(size):
                if board._tiles[x][y].colour is None:
                    moves.append((x, y))
        return moves

    def _clean_json_text(self, text: str) -> str:
        """Strip code fences etc. to get raw JSON."""
        text = text.strip()
        if text.startswith("```"):
            # Remove ```json ... ``` wrapper
            text = text.strip("`")
            lines = text.splitlines()
            if lines and lines[0].strip().lower().startswith("json"):
                lines = lines[1:]
            text = "\n".join(lines)
        return text.strip()

    def _ask_llm(self, prompt: str) -> tuple[int, int] | None:
        """Send prompt, extract (x, y) or None on failure."""
        try:
            resp = _model.generate_content(prompt)
            text = resp.text or ""
            text = self._clean_json_text(text)
        except Exception:
            return None

        try:
            move = json.loads(text)
            return int(move["x"]), int(move["y"])
        except Exception:
            return None

    def _is_valid_move(self, board: Board, move: Move) -> bool:
        """Check move is on board and empty."""
        size = len(board._tiles)
        if not (0 <= move.x < size and 0 <= move.y < size):
            return False
        return board._tiles[move.x][move.y].colour is None


    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if turn == 2:
            return Move(-1, -1)

        board_text = self._board_to_text(board)
        legal_moves = self._legal_moves(board)

        # If no moves, fail gracefully
        if not legal_moves:
            # Should never happen in proper Hex game
            x, y = choice(self._choices)
            return Move(x, y)

        legal_str = ", ".join(f"({x},{y})" for x, y in legal_moves)

        base_prompt = (
            f"You are playing Hex as {self.colour.name}.\n"
            f"Board size: {self._board_size}x{self._board_size}.\n"
            "Board representation (R=Red, B=Blue, .=empty):\n"
            f"{board_text}\n\n"
            "Red wins by connecting TOP to BOTTOM.\n"
            "Blue wins by connecting LEFT to RIGHT.\n"
            "It is your turn now.\n"
            f"Legal empty coordinates are exactly this set: [{legal_str}].\n"
            "Choose the move that maximizes your winning chances, "
            "considering connection strength, blocking opponent paths, "
            "and overall long-term strategy.\n\n"
            f"Last opponent move: {opp_move}\n\n"
            "Return ONLY a single JSON object of the form:\n"
            "{\"x\": int, \"y\": int}\n"
            "where (x, y) is one of the legal coordinates above."
        )

        # First attempt
        xy = self._ask_llm(base_prompt)
        if xy is None:
            x, y = choice(legal_moves)
            return Move(x, y)

        x, y = xy
        move = Move(x, y)

        # If invalid, ask again a few times with constraints
        previous_attempts: set[tuple[int, int]] = {(x, y)}
        max_retries = 3
        retries = 0

        while not self._is_valid_move(board, move) and retries < max_retries:
            invalid_str = ", ".join(str(p) for p in previous_attempts)
            retry_prompt = (
                "The following moves are invalid (occupied or out of bounds): "
                f"{invalid_str}.\n"
                "You MUST choose a different move, strictly from the legal set: "
                f"[{legal_str}].\n"
                "Return ONLY JSON: {\"x\": int, \"y\": int}."
            )

            xy = self._ask_llm(base_prompt + "\n\n" + retry_prompt)
            if xy is None:
                break

            x, y = xy
            move = Move(x, y)
            previous_attempts.add((x, y))
            retries += 1

        # Final fallback to a random legal move
        if not self._is_valid_move(board, move):
            x, y = choice(legal_moves)
            move = Move(x, y)

        return move
