import sys

import tetris
from agent import Agent
from tqdm import tqdm


def training(episode_number=10000, render=False) :
    """
    The training of the agent with the tetris environment
    """
    # --- Initialisation --- #
    game = tetris.Tetris(height=10)
    agent = Agent(input_size=4, decay=0.9995)
    saving_weights_each_steps = 1000
    print("\n >>> Begin Epsilon = " + str(agent.epsilon))
    print(" >>> Decay = " + str(agent.decay))

    # -- Episode LOOP -- #
    for i in tqdm(range(episode_number)) :
        # - Game and Board reset - #
        done = False
        game.reset()
        board = game.get_current_board_state()
        previous_state = game.get_state_properties(board)

        while not done :
            # fetch all the next possible states.
            possible_future_states = game.get_next_states()
            # the agent then decide the next action
            action, actual_state = agent.act_train(possible_future_states)

            # Performs the action
            reward, done = game.step(action, render=render)

            # Saves the move in memory
            agent.fill_memory(previous_state, actual_state, reward, done)

            # Resets iteration for the next move
            previous_state = actual_state

        # train the weights of the NN after the episode
        agent.training_montage()

        if i % saving_weights_each_steps == 0:
            agent.save("weights_{}.h5".format(int(i)))
    agent.save("weights.h5")

    print("\n >>> End Epsilon = " +str(agent.epsilon))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        nb_episodes = int(sys.argv[1])
        training(nb_episodes)
    else:
        training()









