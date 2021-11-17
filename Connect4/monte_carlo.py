from bot import Bot
from common import MONTE_CARLO
import random
import math


class MonteCarlo(Bot):
    """
    This class is responsible for the Monte Carlo Tree Search algorithm.
    The principle of the algorithm is to perform a series of selection,
    expansion, random simulation and backpropagation for a number of iterations
    which is a parameter of the algorithm. A reward system is then implemented based
    on an exploration parameter (also a parameter of the algorithm), and the best move
    can be decided based on that. The more iterations, the better the result, but the
    slower the execution.
    """

    def __init__(self, game, iteration):
        super().__init__(game, bot_type=MONTE_CARLO, iteration=iteration)

    def monte_carlo_tree_search(self, iterations, root, exploration_parameter):
        """
        Main function of MCTS, called whenever a move is needed.

        :param iterations: number of iterations the MCTS algorithm will run for
            (the more iterations the longer the algorithm takes)
        :param root: root tree, starting point of the algorithm (board of the game
            at the moment a move is wanted)
        :param exploration_parameter: factor used in the MCTS

        :return: column where to place the piece
        """
        for i in range(iterations):
            node, turn = self.selection(root, 1, exploration_parameter)
            reward = self.simulation(node.state, turn)
            self.backpropagation(node, reward, turn)

        ans = self.best_child(root, 0)
        return ans.state.last_move[0]

    def selection(self, node, turn, exploration_parameter):
        """
        Expands the root node and takes the best child everytime until a winning state
        is reached. If a node is not fully explored, it is expanded and a child is returned.
        If it is fully explored, the best child of that node is taken.

        :param node: starting node
        :param turn: -1 or 1 according to which player plays next
        :param exploration_parameter: factor used for the MCTS algorithm
        """
        while not node.state.last_move or not node.state.check_win(
            node.state.last_move
        ):
            if not node.fully_explored():
                return self.expansion(node), -1 * turn
            else:
                node = self.best_child(node, exploration_parameter)
                turn *= -1

        return node, turn

    def expansion(self, node):
        """
        Add a child state to the node. Concretely, plays a move and adds
        it to the board of a newly created state, which is the child of the
        current node.

        :param node: current node to expand
        :return: a newly created child of the current node
        """
        free_cols = node.state.get_valid_locations()

        for col in free_cols:
            if col not in node.children_moves:
                new_state = node.state.copy_state()
                new_state.place(col)
                break

        node.add_child(new_state, col)
        return node.children[-1]

    def simulation(self, state_init, turn):
        """
        Simulates random moves until the game is won by someone and returns a reward.
        Until a winning (or losing) situation is obtained, random moves are performed.
        The reward is then simply 1 in case the winner is the actual player, and -1 otherwise.

        :param state_init: current state from which we should end up finding a winning situation
        :param turn: 1 or -1 depending on whose turn it is

        :return: a reward
        """
        state = state_init.copy_state()
        while not state.last_move or not state.check_win(state.last_move):
            free_cols = state.get_valid_locations()
            col = random.choice(free_cols)
            state.place(col)
            turn *= -1

        reward_bool = state.check_win(state.last_move)
        if reward_bool and turn == -1:
            reward = 1
        elif reward_bool and turn == 1:
            reward = -1
        else:
            reward = 0
        return reward

    def backpropagation(self, node, reward, turn):
        """
        Update the rewards of all the ancestors of a node. The reward is sometimes
        added and sometimes substracted from the current reward, since it takes
        into account the fact a winning move from the current player should be
        encouraged, but a winning move from the opponent should be discouraged.

        :param node: current node from which we start the backtracking
        :param reward: reward corresponding to that particular node
        :param turn: 1 or -1 depending on whose turn it is
        """
        while node != None:
            node.visits += 1
            node.reward -= turn * reward
            # node.reward += reward
            node = node.parent
            turn *= -1
        return

    def best_child(self, node, exploration_parameter):
        """
        Returns the best child of a node based on a scoring system proposed by Auer,
        Cesa-Bianchi and Fischer. This formula combines a term of exploration and
        a term of exploitation.

        :param node: node from which we want to find the best child
        :param factor: exploration parameter
        """
        best_score = -float("inf")
        best_children = []
        for c in node.children:
            exploitation = c.reward / c.visits
            exploration = math.sqrt(math.log2(node.visits) / c.visits)
            score = exploitation + exploration_parameter * exploration
            if score == best_score:
                best_children.append(c)
            elif score > best_score:
                best_children = [c]
                best_score = score
        res = random.choice(best_children)
        return res
