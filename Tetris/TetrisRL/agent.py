from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import numpy as np
import random


class Agent:
    """
    The Deep Q-learning agent.
    It will interact with the Tetris environment
    """

    def __init__(
        self,
        input_size=4,
        epsilon=0.9,
        decay=0.9995,
        gamma=0.95,
        loss_fct="mse",
        opt_fct="adam",
        mem=2000,
        metrics=None,
        epsilon_min=0.1,
    ):
        """

        :param input_size: number of features given to the NN.
        :param epsilon: Parameter controlling the exploration/exploitation balance.
        :param decay: The epsilon value will decay after each episode by a certain value. This parameter defines the rate.
        :param gamma: This is the discount factor in the Bellman equation
        :param loss_fct: Functions which will calculates the error obtained for the NN predictions.
        :param opt_fct: Optimization function for the NN
        :param mem: Memory size of the past experiences. By default 2000
        :param metrics: Those are the metrics monitored during the training phase of the neural networks
        :param epsilon_min: This is the lowest value possible for the epsilon parameter
        """
        if metrics is None:
            metrics = ["mean_squared_error"]
        self.input_size = input_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.loss_fct = loss_fct
        self.opt_fct = opt_fct
        self.memory = deque(maxlen=mem)
        self.decay = decay
        self.metrics = metrics
        self.epsilon_min = epsilon_min
        # build the neural network
        self.model = Sequential()
        self.model.add(Dense(64, activation="relu", input_shape=(input_size,)))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(
            optimizer=self.opt_fct, loss=self.loss_fct, metrics=self.metrics
        )

    def act_train(self, states):
        """
        This function returns the best of multiple states or a random one. A random state is returned if a random
        percentage is less than the current epsilon
        :param states: next possible states
        :return: the best state or a random state
        """
        # select a random state with a random probability
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(states)

        best_score = None
        best_state = None
        # compute the Q-value for each of the next state and select the best
        scores = self._predict_scores([state for action, state in states])
        for i, (action, state) in enumerate(states):
            score = scores[i]
            if not best_score or score > best_score:
                best_score = score
                best_state = (action, state)

        return best_state

    def act_best(self, states):
        """
        Same function than act_train, however, no random states are chosen only the best one.
        :param states: next possible states
        :return: the best state
        """
        best_score = None
        best_state = None
        # compute the Q-value for each of the next state and select the best
        scores = self._predict_scores([state for action, state in states])
        for i, (action, state) in enumerate(states):
            score = scores[i]
            if not best_score or score > best_score:
                best_score = score
                best_state = (action, state)

        return best_state

    def _predict_scores(self, states):
        """
        Return the score for a list of states
        :param states: input list of states
        :return: list of q-values for each state
        """
        input = np.array(states)
        predictions = self.model.predict(input)
        return [prediction[0] for prediction in predictions]

    def fill_memory(self, previous_state, next_state, reward, done):
        """
        Fill the buffer with previous experiences
        :param previous_state:original state
        :param next_state:state chosen by the network
        :param reward:reward received
        :param done:boolean value to signify whether the end of an episode is reached
        """
        self.memory.append((previous_state, next_state, reward, done))

    def save(self, path: str):
        """
        save the weights of the network
        :param path: filepath where weights are saved
        """
        self.model.save_weights(path)

    def load(self, path: str):
        """
        load the weights of the network
        :param path: filepath where weights are saved
        """
        self.model.load_weights(path)

    def training_montage(self, batch_size=64, epochs=1):
        """
        Train the model and adapt its weights using the recent experiences
        :param batch_size: Number of samples used for the training
        :param epochs: number of iterations of the backpropagation
        """
        if len(self.memory) < batch_size:
            return
        # Randomly select a batch of experiences
        experiences = random.sample(self.memory, batch_size)

        # compute the target for the neural network
        next_states = [experience[1] for experience in experiences]
        scores = self._predict_scores(next_states)
        dataset = []
        target = []
        for i in range(batch_size):
            previous_state, _, reward, done = experiences[i]
            if not done:
                next_q = self.gamma * scores[i] + reward
            else:
                next_q = reward
            dataset.append(previous_state)
            target.append(next_q)
        self.model.fit(
            dataset, target, batch_size, epochs, verbose=0
        )  # train the model
        self.epsilon = max(
            self.epsilon * self.decay, self.epsilon_min
        )  # explore less


if __name__ == "__main__":
    dqn = Agent()
    dqn.model.summary()
