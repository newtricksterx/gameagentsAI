from typing import List, Any

import numpy as np
from copy import deepcopy

from numpy import ndarray

from agents.agent import Agent
from store import register_agent
from agents.minimax_agent import evaluate
from agents.minimax_agent import generate_possible_moves
import time


def get_possible_moves(max_step, my_pos, chess_board, adv_pos):
    """
    Returns all possible moves at a position.
    Returns in format [x, y coordinate, move direction] = [x, y, dir]
    """
    # List of moves
    l = []
    width = 0
    # Iterate over a "diamond" pattern around the position:
    for i in range(-max_step, max_step + 1):
        r = my_pos[0]
        r += i
        for k in range(-width, width + 1):
            c = my_pos[1]
            c += k
            if adv_pos[0] == r and adv_pos[1] == c:
                # Cannot move into adversary position, move not considered
                continue
            if r < len(chess_board[0]) and c < len(chess_board[1]) and r >= 0 and c >= 0:
                # If in range of board, check all directions to place a wall
                for j in range(4):
                    if not chess_board[r][c][j]:
                        if check_valid_step(chess_board, adv_pos, my_pos, (r, c), j, max_step):
                            l.append([(r, c), j])
        if i < 0:
            width += 1
        else:
            width -= 1
    return l


def check_valid_step(chess_board, adv_pos, start_pos, end_pos, barrier_dir, max_step):
    """ Checks if a move is valid (modified from world.py) """
    # Endpoint already has barrier or is boarder
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    r, c = end_pos
    if chess_board[r, c, barrier_dir]:
        return False
    if np.array_equal(start_pos, end_pos):
        return True
    # BFS
    state_queue = [(start_pos, 0)]
    visited = {tuple(start_pos)}
    is_reached = False
    while state_queue and not is_reached:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        if cur_step == max_step:
            break
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue

            next_pos = cur_pos[0] + move[0], cur_pos[1] + move[1]
            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue
            if np.array_equal(next_pos, end_pos):
                is_reached = True
                break

            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

    return is_reached


def check_endgame(chess_board, my_pos, adv_pos):
    """
    Check if the simulation ends. (modified from world.py)
    Returns the score of the player at my_pos if the game ends.
    """
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    # Union-Find
    father = dict()
    for r in range(len(chess_board[0])):
        for c in range(len(chess_board[0])):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(len(chess_board[0])):
        for c in range(len(chess_board[0])):
            for direction, move in enumerate(
                    moves[1:3]
            ):  # Only check down and right
                if chess_board[r, c, direction + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(len(chess_board[0])):
        for c in range(len(chess_board[0])):
            find((r, c))
    p0_r = find(tuple(my_pos))
    p1_r = find(tuple(adv_pos))
    p0_score = list(father.values()).count(p0_r)
    if p0_r == p1_r:
        return False, p0_score

    return True, p0_score


def run_one_simulation(chess_board, adv_pos, max_step, move) -> int:
    """ Runs one random simulation of the game and returns the score. Initial state of game specified by caller. """
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    board = deepcopy(chess_board)  # Don't want to modify the original chessboard

    adv_pos_c = adv_pos
    # Apply the predetermined move:
    my_pos_c = move[0][0], move[0][1]
    board[my_pos_c[0], my_pos_c[1], move[1]] = True
    board[my_pos_c[0] + moves[move[1]][0], my_pos_c[1] + moves[move[1]][1], (move[1] + 2) % 4] = True

    # Run random moves until simulated game finishes:
    while True:
        # Check if game is done
        final_score = evaluate(board, my_pos_c, adv_pos_c)
        if final_score != -1:
            return final_score

        # Random step for player 1:
        p1_step = random_step(board, my_pos_c, adv_pos_c, max_step)
        r1, c1, d1 = p1_step
        my_pos_c = r1, c1
        board[r1, c1, d1] = True
        board[r1 + moves[d1][0], c1 + moves[d1][1], (d1 + 2) % 4] = True

        final_score = evaluate(board, my_pos_c, adv_pos_c)
        if final_score != -1:
            return final_score

        # Random step for player 2:
        p2_step = random_step(board, adv_pos_c, my_pos_c, max_step)
        r2, c2, d2 = p2_step
        adv_pos_c = r2, c2
        board[r2, c2, d2] = True
        board[r2 + moves[d2][0], c2 + moves[d2][1], (d2 + 2) % 4] = True


def random_step(board, my_pos, adv_pos, max_step):
    """
    Performs one random step for an agent. Used for the monte carlo "inner" simulations.
    Algorithm obtained from given class random_agent.
    """
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    steps = np.random.randint(0, max_step + 1)
    # Random Walk
    for _ in range(steps):
        ori_pos = deepcopy(my_pos)
        r, c = my_pos
        direction = np.random.randint(0, 4)
        m_r, m_c = moves[direction]
        my_pos = (r + m_r, c + m_c)

        # Special Case enclosed by Adversary
        k = 0
        while board[r, c, direction] or my_pos == adv_pos:
            k += 1
            if k > 300:
                break
            direction = np.random.randint(0, 4)
            m_r, m_c = moves[direction]
            my_pos = (r + m_r, c + m_c)

        if k > 300:
            my_pos = ori_pos
            break

    # Put down Barrier
    direction = np.random.randint(0, 4)
    r, c = my_pos
    while board[r, c, direction]:
        direction = np.random.randint(0, 4)
    return [r, c, direction]


def rolling_average(old_avg, new_val, n):
    """ Helper function to get an average from a previous average and a new value """
    return old_avg*((n-1)/n) + new_val/n


@register_agent("montecarlo_agent")
class MonteCarloAgent(Agent):
    """
    An agent that implements a pure monte carlo algorithm with random agents
    """

    def __init__(self):
        super(MonteCarloAgent, self).__init__()
        self.name = "MonteCarloAgent"
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """ Step function for the Monte Carlo Algorithm """
        start_time = time.time()

        best_move = [0, 0]  # index, score
        moves_list = get_possible_moves(max_step, my_pos, chess_board, adv_pos)
        # moves_list = get_moves(max_step, my_pos, chess_board, adv_pos)
        # print("moves_list:", moves_list)
        moves_gain = [0] * len(moves_list)
        i, j = 0, 1
        b = False
        while True:  # Run simulations until we run out of time (2s limit)
            for i in range(len(moves_list)):
                if (time.time() - start_time) >= 1.98:
                    # Time is about to go over the limit, break
                    b = True
                    break
                moves_gain[i] = rolling_average(
                    moves_gain[i],
                    run_one_simulation(chess_board, adv_pos, max_step, moves_list[i]),
                    j)
                if moves_gain[i] > best_move[1]:
                    best_move = [i, moves_gain[i]]
            if b:
                end_time = time.time()
                break
            j += 1

        # print("moves_gain:", moves_gain)
        # print("MONTE CARLO: Execution time for one step with", i, "simulations per step:", end_time - start_time)
        # print("Maximum gain from simulations found:", moves_list[best_move[0]], "Score:", best_move[1])
        return moves_list[best_move[0]]