# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from agents.minimax_agent import MinimaxAgent
from agents.montecarlo_agent import MonteCarloAgent
import sys
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

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
        # Generate minimax tree for 30s, use minimax results for moves until we run into leaf node
        # after, proceed with monte carlo

        minimax = MinimaxAgent()
        montecarlo = MonteCarloAgent()
        start_time = time.time()

        if len(chess_board[0]) < 7:
            try:
                return MinimaxAgent.step(minimax, chess_board, my_pos, adv_pos, max_step)
            except:
                return MonteCarloAgent.step(montecarlo, chess_board, my_pos, adv_pos, max_step)
        else:
            return MonteCarloAgent.step(montecarlo, chess_board, my_pos, adv_pos, max_step)
