# Minimax agent
import gc
import time
from agents.agent import Agent
from store import register_agent
#from agents.montecarlo_agent import get_possible_moves
from copy import deepcopy
import numpy as np
import sys

global chess_board_node_database
chess_board_node_database = []


def update_chess_board(chess_board, move, barrier_dir):
    chess_board[move[0], move[1], barrier_dir] = True

    if barrier_dir == 0 and within_boundary(chess_board, (move[0] - 1, move[1])):
        chess_board[move[0] - 1, move[1], 2] = True

    elif barrier_dir == 2 and within_boundary(chess_board, (move[0] + 1, move[1])):
        chess_board[move[0] + 1, move[1], 0] = True

    elif barrier_dir == 1 and within_boundary(chess_board, (move[0], move[1] + 1)):
        chess_board[move[0], move[1] + 1, 3] = True

    elif barrier_dir == 3 and within_boundary(chess_board, (move[0], move[1] - 1)):
        chess_board[move[0], move[1] - 1, 1] = True


def within_boundary(chess_board, pos):
    boundary = (chess_board[0].size / 4) - 1
    if boundary >= pos[0] >= 0 and boundary >= pos[1] >= 0:
        return True

    return False


def check_valid_move(chess_board, my_pos, to_pos, adv_pos, max_steps):
    dir_map = {
        "u": 0,
        "r": 1,
        "d": 2,
        "l": 3,
    }

    if my_pos == to_pos:
        return True

    if to_pos == adv_pos:
        return False

    # checks if the total number of row steps and column steps is not greater than max_steps
    if abs(my_pos[0] - to_pos[0]) + abs(my_pos[1] - to_pos[1]) > max_steps:
        # print("too many steps")
        return False

    # check if path exists to new move
    queue = [(my_pos, 0)]
    visited = [my_pos]


    while len(queue) > 0:
        pos, steps = queue.pop()
        next_step = steps + 1
        if pos == to_pos:
            return True
        if steps == max_steps:
            continue
        elif steps > max_steps:
            return False

        if not chess_board[pos[0], pos[1], dir_map["u"]] and within_boundary(chess_board, (pos[0] - 1, pos[1])):
            new_pos = (pos[0] - 1, pos[1])
            if new_pos not in visited:
                queue.append((new_pos, next_step))
                visited.append(new_pos)

        if not chess_board[pos[0], pos[1], dir_map["d"]] and within_boundary(chess_board, (pos[0] + 1, pos[1])):
            new_pos = (pos[0] + 1, pos[1])
            if new_pos not in visited:
                queue.append((new_pos, next_step))
                visited.append(new_pos)

        if not chess_board[pos[0], pos[1], dir_map["l"]] and within_boundary(chess_board, (pos[0], pos[1] - 1)):
            new_pos = (pos[0], pos[1] - 1)
            if new_pos not in visited:
                queue.append((new_pos, next_step))
                visited.append(new_pos)

        if not chess_board[pos[0], pos[1], dir_map["r"]] and within_boundary(chess_board, (pos[0], pos[1] + 1)):
            new_pos = (pos[0], pos[1] + 1)
            if new_pos not in visited:
                queue.append((new_pos, next_step))
                visited.append(new_pos)

    return False


def generate_possible_moves(chess_board, my_pos, adv_pos, max_steps, max_min):
    possible_moves = []
    possible_moves_dir = []
    r, c = my_pos
    if max_min == "min":
        r, c = adv_pos

    for row_steps in range(max_steps + 1):
        for col_steps in range(max_steps - row_steps + 1):
            new_pos = (r + row_steps, c + col_steps)

            if new_pos not in possible_moves and within_boundary(chess_board, new_pos):
                possible_moves.append(new_pos)

            new_pos = (r - row_steps, c - col_steps)

            if new_pos not in possible_moves and within_boundary(chess_board, new_pos):
                possible_moves.append(new_pos)

            new_pos = (r + row_steps, c - col_steps)

            if new_pos not in possible_moves and within_boundary(chess_board, new_pos):
                possible_moves.append(new_pos)

            new_pos = (r - row_steps, c + col_steps)
            if new_pos not in possible_moves and within_boundary(chess_board, new_pos):
                possible_moves.append(new_pos)

    for move in possible_moves:

        if not check_valid_move(chess_board, my_pos, move, adv_pos, max_steps):
            continue

        # print(move)
        if not chess_board[move[0], move[1], 0] and (move, 0) not in possible_moves_dir:
            possible_moves_dir.append((move, 0))

        if not chess_board[move[0], move[1], 1] and (move, 1) not in possible_moves_dir:
            possible_moves_dir.append((move, 1))

        if not chess_board[move[0], move[1], 2] and (move, 2) not in possible_moves_dir:
            possible_moves_dir.append((move, 2))

        if not chess_board[move[0], move[1], 3] and (move, 3) not in possible_moves_dir:
            possible_moves_dir.append((move, 3))

    return possible_moves_dir


# will return -1 if not at a goal state
# will return a number > 0 if yes at a goal state
def evaluate(chess_board, my_pos, adv_pos):
    ori_pos = deepcopy(my_pos)
    queue = [ori_pos]
    visited = [ori_pos]
    count = 1

    while len(queue) > 0:
        pos = queue.pop()
        if pos == adv_pos:
            return -1

        if not chess_board[pos[0], pos[1], 0]:
            new_pos = (pos[0] - 1, pos[1])
            if within_boundary(chess_board, new_pos) and new_pos not in visited:
                queue.append(new_pos)
                visited.append(new_pos)
                count += 1

        if not chess_board[pos[0], pos[1], 1]:
            new_pos = (pos[0], pos[1] + 1)
            if within_boundary(chess_board, new_pos) and new_pos not in visited:
                queue.append(new_pos)
                visited.append(new_pos)
                count += 1

        if not chess_board[pos[0], pos[1], 2]:
            new_pos = (pos[0] + 1, pos[1])
            if within_boundary(chess_board, new_pos) and new_pos not in visited:
                queue.append(new_pos)
                visited.append(new_pos)
                count += 1

        if not chess_board[pos[0], pos[1], 3]:
            new_pos = (pos[0], pos[1] - 1)
            if within_boundary(chess_board, new_pos) and new_pos not in visited:
                queue.append(new_pos)
                visited.append(new_pos)
                count += 1

    return count


def expected_value(chess_board, my_pos, adv_pos):
    queue = [my_pos]
    visited = [my_pos]
    count = 1

    can_go_u = True
    can_go_d = True
    can_go_l = True
    can_go_r = True

    while len(queue) > 0:
        pos = queue.pop()
        if pos == adv_pos:
            break

        if chess_board[pos[0], pos[1], 0]:
            can_go_u = False
        else:
            new_pos = (pos[0] - 1, pos[1])
            if new_pos not in visited and within_boundary(chess_board, new_pos) and can_go_u:
                queue.append(new_pos)
                visited.append(new_pos)
                count += 1

        if chess_board[pos[0], pos[1], 1]:
            can_go_r = False
        else:
            new_pos = (pos[0], pos[1] + 1)
            if new_pos not in visited and within_boundary(chess_board, new_pos) and can_go_r:
                queue.append(new_pos)
                visited.append(new_pos)
                count += 1

        if chess_board[pos[0], pos[1], 2]:
            can_go_d = False
        else:
            new_pos = (pos[0] + 1, pos[1])
            if new_pos not in visited and within_boundary(chess_board, new_pos) and can_go_d:
                queue.append(new_pos)
                visited.append(new_pos)
                count += 1

        if chess_board[pos[0], pos[1], 3]:
            can_go_d = False
        else:
            new_pos = (pos[0], pos[1] - 1)
            if new_pos not in visited and within_boundary(chess_board, new_pos) and can_go_l:
                queue.append(new_pos)
                visited.append(new_pos)
                count += 1

    return count

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

def gen_neighbour_nodes(node, max_steps):
    neighbour_nodes = []
    next_max_min = "max"

    if node.max_min == "max":
        next_max_min = "min"

    #all_possible_moves = generate_possible_moves(node.chess_board, node.my_pos, node.adv_pos, max_steps, node.max_min)
    all_possible_moves = get_possible_moves(max_steps, node.my_pos, node.chess_board, node.adv_pos)
    #print(all_possible_moves)

    for move_dir in all_possible_moves:
        new_chess_board = deepcopy(node.chess_board)
        move, barrier_dir = move_dir
        update_chess_board(new_chess_board, move, barrier_dir)

        new_node = Node(node, None, -1, new_chess_board, move, node.adv_pos, barrier_dir, next_max_min, node.depth + 1)
        neighbour_nodes.append(new_node)

        if new_node not in chess_board_node_database:
            chess_board_node_database.append(new_node)

    return neighbour_nodes


def minimax(root):
    stack = [root]
    visited = []

    while len(stack) > 0:
        node = stack.pop()

        if not check_neighbours_all_visited(node, visited):
            stack.append(node)
            for neighbour in node.neighbours:
                stack.append(neighbour)

        else:
            propagate_utility(node)
            visited.append(node)


def propagate_utility(node):
    if node.utility != -1 and node.parent is not None:
        if node.parent.max_min == "max":
            if node.parent.utility == -1 or node.utility > node.parent.utility:
                node.parent.utility = node.utility
        elif node.parent.max_min == "min":
            if node.parent.utility == -1 or node.utility < node.parent.utility:
                node.parent.utility = node.utility


def check_neighbours_all_visited(node, visited):
    if node.neighbours is None:
        return True

    for neighbours in node.neighbours:
        if neighbours not in visited:
            return False

    return True


def gen_next_step(chess_board, my_pos, adv_pos, max_steps, first_step):
    root = Node(None, None, -1, chess_board, my_pos, adv_pos, None, "max", 0)

    if root not in chess_board_node_database:
        chess_board_node_database.append(root)

    queue = [root]

    a = time.time()

    while len(queue) > 0:
        cur_node = queue.pop()
        cur_depth = cur_node.depth

        if cur_depth < 10:
            all_neighbours = gen_neighbour_nodes(cur_node, max_steps)
            cur_node.neighbours = all_neighbours
            if all_neighbours is not []:
                for neighbours in all_neighbours:
                    queue.append(neighbours)
            else:
                cur_node.utility = evaluate(chess_board, cur_node.my_pos, cur_node.adv_pos)
                if cur_node.utility == -1:
                    cur_node.utility = expected_value(chess_board, my_pos, adv_pos)
        else:
            # cur_node.utility = expected_win_loss(chess_board, cur_node.my_pos, cur_node.adv_pos, max_steps, 10)
            cur_node.utility = expected_value(chess_board, my_pos, adv_pos)

    b = time.time()

    print("Tree gen time: ", b - a)

    a = time.time()

    minimax(root)

    b = time.time()

    print("minimax time taken: ", b - a)

    return get_next_step_helper(root)


def get_next_step_helper(root):
    move = root.my_pos
    barrier_dir = 0


    for neighbours_node in root.neighbours:
        if neighbours_node.utility == root.utility:
            move = neighbours_node.my_pos
            barrier_dir = neighbours_node.barrier_dir

    print(root.utility)

    return move, barrier_dir


class Node:
    def __init__(self, parent, neighbours, utility, chess_board, my_pos, adv_pos, barrier_dir, max_min, depth):
        self.parent = parent  # Node object
        self.neighbours = neighbours  # Array of Node Objects
        self.utility = utility  # integer (default = -1)
        self.chess_board = chess_board  # nump.ndarray
        self.my_pos = my_pos  # tuple (row, col)
        self.adv_pos = adv_pos  # tuple (row, col)
        self.barrier_dir = barrier_dir  # integer (0 - 3)
        self.max_min = max_min  # node max/min player
        self.depth = depth  # depth in tree


@register_agent("minimax_agent")
class MinimaxAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MinimaxAgent, self).__init__()
        self.name = "MinimaxAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True

    """
    Main decision logic of the agent, which is called by the simulator.
    Extend this method to implement your own agent to play the game.

    Parameters
    ----------
    chess_board : numpy.ndarray of shape (board_size, board_size, 4)
        The chess board.
    my_pos : tuple of int
        The position of the agent.
    adv_pos : tuple of int
        The position of the adversary (opponent).
    max_step : int
        The maximum number of steps that the agent can take.

    Returns
        -------
    my_pos : tuple of int
        The new position of the agent.
    dir : int
         The direction of the agent, as defined in world.py (DIRECTION_UP/DIRECTION_DOWN/DIRECTION_LEFT/DIRECTION_RIGHT)
    """

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # print(len(chess_board_node_database))

        for node in chess_board_node_database:
            if np.array_equal(chess_board, node.chess_board) and node.my_pos == my_pos and node.adv_pos == adv_pos \
                    and node.utility != -1:
                return get_next_step_helper(node)

        s = time.time()

        if len(chess_board_node_database) == 0:
            r = gen_next_step(chess_board, my_pos, adv_pos, max_step, False)
        else:
            raise Exception

        e = time.time()

        print("Time elapsed: ", e - s)

        return r
