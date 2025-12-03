import argparse
import importlib
import sys
import os
import concurrent.futures
from contextlib import redirect_stdout, redirect_stderr

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

# 0 = alternate starts
# 1 = player1 always starts
# 2 = player2 always starts
START_RULE = 0
game_counter = 0


def run_single_game(
    p1_path: str,
    p1_class: str,
    p1_name: str,
    p2_path: str,
    p2_class: str,
    p2_name: str,
    board_size: int,
    verbose: bool,
    silent: bool,
) -> dict[str, str] | None:
    """
    Helper to run one game in isolation.
    """

    # Import modules fresh inside the worker
    p1_module = importlib.import_module(p1_path)
    p2_module = importlib.import_module(p2_path)

    # Decide who goes first
    global game_counter

    if START_RULE == 1:
        first = 1
    elif START_RULE == 2:
        first = 2
    else:  # START_RULE == 0
        first = 1 if (game_counter % 2 == 0) else 2

    game_counter += 1

    # Assign RED (first) and BLUE (second)
    if first == 1:
        red_player = Player(
            name=p1_name,
            agent=getattr(p1_module, p1_class)(Colour.RED),
        )
        blue_player = Player(
            name=p2_name,
            agent=getattr(p2_module, p2_class)(Colour.BLUE),
        )
    else:
        red_player = Player(
            name=p2_name,
            agent=getattr(p2_module, p2_class)(Colour.RED),
        )
        blue_player = Player(
            name=p1_name,
            agent=getattr(p1_module, p1_class)(Colour.BLUE),
        )

    # Create game
    g = Game(
        player1=red_player,
        player2=blue_player,
        board_size=board_size,
        logDest=sys.stderr,
        verbose=verbose,
        silent=silent,
    )

    # Run silently
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        result = g.run()

    return result


def print_progress(
    completed: int,
    total: int,
    p1_name: str,
    p1_wins: int,
    p2_name: str,
    p2_wins: int,
) -> None:
    """Print a single-line progress display with two win bars (one per agent)."""
    bar_length = 20  # length of each agent's bar

    # Scale bars by total number of games
    if total > 0:
        p1_fill = int(bar_length * (p1_wins / total))
        p2_fill = int(bar_length * (p2_wins / total))
    else:
        p1_fill = p2_fill = 0

    p1_bar = "#" * p1_fill + "-" * (bar_length - p1_fill)
    p2_bar = "#" * p2_fill + "-" * (bar_length - p2_fill)

    # Win rates based on completed games so far
    p1_rate = (p1_wins / completed * 100.0) if completed > 0 else 0.0
    p2_rate = (p2_wins / completed * 100.0) if completed > 0 else 0.0

    line = (
        f"\r"
        f"{p1_name:10s} [{p1_bar}] {p1_wins:3d} ({p1_rate:5.1f}%)   "
        f"{p2_name:10s} [{p2_bar}] {p2_wins:3d} ({p2_rate:5.1f}%)   "
        f"{completed}/{total}"
    )
    print(line, end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Hex",
        description="Run a game of Hex. By default, two naive agents will play.",
    )
    parser.add_argument(
        "-p1",
        "--player1",
        default="agents.DefaultAgents.NaiveAgent NaiveAgent",
        type=str,
        help=(
            "Specify the player 1 agent, format: "
            "agents.GroupX.AgentFile AgentClassName .e.g. "
            "agents.Group0.NaiveAgent NaiveAgent"
        ),
    )
    parser.add_argument(
        "-p1Name",
        "--player1Name",
        default="Red",
        type=str,
        help="Specify the player 1 name",
    )
    parser.add_argument(
        "-p2",
        "--player2",
        default="agents.DefaultAgents.NaiveAgent NaiveAgent",
        type=str,
        help=(
            "Specify the player 2 agent, format: "
            "agents.GroupX.AgentFile AgentClassName .e.g. "
            "agents.Group0.NaiveAgent NaiveAgent"
        ),
    )
    parser.add_argument(
        "-p2Name",
        "--player2Name",
        default="Blue",
        type=str,
        help="Specify the player 2 name",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-b",
        "--board_size",
        type=int,
        default=11,
        help="Specify the board size",
    )
    parser.add_argument(
        "-l",
        "--log",
        nargs="?",
        type=str,
        default=sys.stderr,
        const="game.log",
        help=(
            "Save moves history to a log file,"
            "if the flag is present, the result will be saved to game.log."
            "If a filename is provided, the result will be saved to the provided file."
            "If the flag is not present, the result will be printed to the console, via stderr."
        ),
    )
    parser.add_argument(
        "-n",
        "--num_games",
        type=int,
        default=1,
        help="Number of games to run to estimate win rates.",
    )

    args = parser.parse_args()
    p1_path, p1_class = args.player1.split(" ")
    p2_path, p2_class = args.player2.split(" ")

    # Single-game mode: keep behaviour as before
    if args.num_games == 1:
        # original single-game behaviour
        p1_module = importlib.import_module(p1_path)
        p2_module = importlib.import_module(p2_path)
        g = Game(
            player1=Player(
                name=args.player1Name,
                agent=getattr(p1_module, p1_class)(Colour.RED),
            ),
            player2=Player(
                name=args.player2Name,
                agent=getattr(p2_module, p2_class)(Colour.BLUE),
            ),
            board_size=args.board_size,
            logDest=args.log,
            verbose=args.verbose,
            silent=False,
        )
        g.run()
    else:
        # -------- MULTI-GAME SEQUENTIAL MODE --------
        total_games = args.num_games

        # win counts
        p1_wins = 0
        p2_wins = 0
        other_results = 0    # abnormal / failed games

        # aggregated timing and turns
        total_p1_move_time = 0.0   # sum of player1_move_time (seconds)
        total_p2_move_time = 0.0   # sum of player2_move_time (seconds)
        total_p1_turns = 0         # sum of player1_turns
        total_p2_turns = 0         # sum of player2_turns
        total_game_time = 0.0      # sum of total_time for all games
        games_counted = 0          # number of games with valid stats

        # Print initial progress bar
        print_progress(
            0,
            total_games,
            args.player1Name,
            p1_wins,
            args.player2Name,
            p2_wins,
        )

        for i in range(total_games):
            result = run_single_game(
                p1_path,
                p1_class,
                args.player1Name,
                p2_path,
                p2_class,
                args.player2Name,
                args.board_size,
                False,  # verbose off for batch
                True,   # silent game output
            )

            if result is None:
                other_results += 1
            else:
                winner = result.get("winner")

                # Names from the result (these are the *slots* in Game, not CLI positions)
                res_p1_name = result.get("player1")
                res_p2_name = result.get("player2")

                # ---- wins (by *actual* CLI name) ----
                if winner == args.player1Name:
                    p1_wins += 1
                elif winner == args.player2Name:
                    p2_wins += 1
                else:
                    other_results += 1

                # ---- timing & turns (assign by name) ----
                try:
                    res_p1_time = float(result.get("player1_move_time", 0.0))
                    res_p2_time = float(result.get("player2_move_time", 0.0))
                    res_p1_turns = int(result.get("player1_turns", 0))
                    res_p2_turns = int(result.get("player2_turns", 0))
                    game_time = float(result.get("total_game_time", 0.0))
                except (TypeError, ValueError):
                    other_results += 1
                else:
                    # player1 slot in result
                    if res_p1_name == args.player1Name:
                        total_p1_move_time += res_p1_time
                        total_p1_turns += res_p1_turns
                    elif res_p1_name == args.player2Name:
                        total_p2_move_time += res_p1_time
                        total_p2_turns += res_p1_turns

                    # player2 slot in result
                    if res_p2_name == args.player1Name:
                        total_p1_move_time += res_p2_time
                        total_p1_turns += res_p2_turns
                    elif res_p2_name == args.player2Name:
                        total_p2_move_time += res_p2_time
                        total_p2_turns += res_p2_turns

                    total_game_time += game_time
                    games_counted += 1

            # progress bar after each game
            print_progress(
                i + 1,
                total_games,
                args.player1Name,
                p1_wins,
                args.player2Name,
                p2_wins,
            )

        print()  # finish progress line
        print()

        # -------- SUMMARY TABLE --------
        if games_counted > 0:
            # Global average total game time (wall-clock)
            avg_game_time = total_game_time / games_counted

            # Win rates (over all attempted games)
            p1_win_rate = (p1_wins / total_games *
                           100.0) if total_games else 0.0
            p2_win_rate = (p2_wins / total_games *
                           100.0) if total_games else 0.0

            # Average thinking time per game
            p1_avg_game_think = total_p1_move_time / games_counted
            p2_avg_game_think = total_p2_move_time / games_counted

            # Average thinking time per move
            p1_avg_move = total_p1_move_time / total_p1_turns if total_p1_turns else 0.0
            p2_avg_move = total_p2_move_time / total_p2_turns if total_p2_turns else 0.0

            p1_avg_move_str = f"{p1_avg_move:.6f}" if total_p1_turns else "n/a"
            p2_avg_move_str = f"{p2_avg_move:.6f}" if total_p2_turns else "n/a"

            print("\nSummary\n")
            print(
                f"{'Player':<12}"
                f"{'Wins':>6}"
                f"{'Win%':>8}"
                f"{'Games':>8}"
                f"{'Avg/game s':>12}"
                f"{'Moves':>8}"
                f"{'Avg/move s':>12}"
            )

            print(
                f"{args.player1Name:<12}"
                f"{p1_wins:>6}"
                f"{p1_win_rate:>8.1f}"
                f"{games_counted:>8}"
                f"{p1_avg_game_think:>12.6f}"
                f"{total_p1_turns:>8}"
                f"{p1_avg_move_str:>12}"
            )

            print(
                f"{args.player2Name:<12}"
                f"{p2_wins:>6}"
                f"{p2_win_rate:>8.1f}"
                f"{games_counted:>8}"
                f"{p2_avg_game_think:>12.6f}"
                f"{total_p2_turns:>8}"
                f"{p2_avg_move_str:>12}"
            )

            print(f"\nOther/failed games: {other_results}")
            print(f"Mean total game time: {avg_game_time:.6f}s")
        else:
            print("\nNo valid games to compute stats.")
