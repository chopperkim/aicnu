import numpy as np
import random
from collections import defaultdict
from environment import Env


# Monte Carlo Agent which learns every episodes from the sample
class MCAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.samples = []
        self.value_table = defaultdict(float)

    # append sample to memory(state, action, reward, done)
    def save_sample(self, state, action, reward, done):
        self.samples.append([state, action, reward, done])

    # for every episode, agent updates q function of visited states
    def update(self, episode):
        G = 0
        visit_state = []
        for state, action, reward, done in reversed(episode):
            state = str(state)
            if state not in visit_state:
                visit_state.append(state)

                # -------- implementation value table update ------------#
                # ------------------- FROM HERE -------------------------#
                # use : 
                #       - G
                #       - self.discount_factor
                #       - self.value_table
                #       - self.learning_rate

                # --------------------UNTIL HERE ------------------------#
                self.value_table[state] = v_new

                

    def get_action(self, state):
        # -------- implementation e-greedy policy ------------#
        # ------------------- FROM HERE -------------------------#
        # use : 
        #      - np.random.rand()
        #      - self.epsilon
        #      - self.possible_next_state_value
        #      - self.arg_max()




        # --------------------UNTIL HERE ------------------------#
        return int(action)

    # compute arg_max if multiple candidates exit, pick one randomly
    @staticmethod
    def arg_max(next_state_value):
        max_index_list = []
        max_value = next_state_value[0]
        for index, value in enumerate(next_state_value):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    # get the possible next state values
    def possible_next_state_value(self, state):
        col, row = state
        next_state_value = [0.0] * 4

        if row != 0:
            next_state_value[0] = self.value_table[str([col, row - 1])]
        else:
            next_state_value[0] = self.value_table[str(state)]
        if row != self.height - 1:
            next_state_value[1] = self.value_table[str([col, row + 1])]
        else:
            next_state_value[1] = self.value_table[str(state)]
        if col != 0:
            next_state_value[2] = self.value_table[str([col - 1, row])]
        else:
            next_state_value[2] = self.value_table[str(state)]
        if col != self.width - 1:
            next_state_value[3] = self.value_table[str([col + 1, row])]
        else:
            next_state_value[3] = self.value_table[str(state)]

        return next_state_value


# utils
def generate_episode(env, agent):
    episode = []
    current_state = env.reset() # init state

    while(True):
        # take random or best action 
        action = agent.get_action(current_state)

        # apply the action to env
        next_state, reward, done = env.step(action, agent.value_table)
        env.render()

        # save episode
        # !!! 중요
        # episode.append( (current_state, action, reward, done) )
        # 로 수행하게 되면, 마지막 hazzard 혹은 goal state 에 대한 value 가 구해지지 않는다.
        # 이것은 다음과 같은 문제를 발생시킨다.
        #
        # 1) 현재 greedy policy + eplison 은 바로 다음 행동을 하게되어 도착할 수 있는 s' 의 값. 즉 v(s') 를 높게 하는
        #    action을 선택한다.
        # 2) hazzard, goal 등의 마지막 땅에 대한 value 값이 계산되지 않으면 영원히 그 땅에 도착하는 action 은
        #    제대로 선택되지 않는다.(random walk 제외)
        # 3) 현재 setting 에서는 init_state 는 항상 고정이기 때문에, init state에 대한 값의 학습은 별로 중요하지 않음에도
        #    계속 학습이 되고 있다.
        #
        # 따라서 간단한 트릭으로, episode 에 현재 state 가 아닌, 도착하게 되는 지점의 state'를 넣어주어 
        # 학습이 제대로 되도록 한다.
        episode.append( (next_state, action, reward, done) )


        if len(episode) > 200: 
            # stop episode for time saving
            return [], False

        if done: 
            break
        
        current_state = next_state   
    return episode, True

# main loop
if __name__ == "__main__":
    env = Env()
    agent = MCAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        print("Episode : ", episode+1)
        current_state = env.reset()

        # generate episode
        episode, _ = generate_episode(env, agent)

        # update value table according to the episode
        agent.update(episode)
            
        # for monitoring values
        env.print_values(agent.value_table)

