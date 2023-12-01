import numpy as np
import gymnasium as gym
from gymnasium import spaces

EPISODES = 10_000
MAX_STEPS = 200
COLUMNS = 10
ROWS = 5
ACTION_MAPPINGS = {
    0: (-1, 0),  # Move up
    1: (1, 0),   # Move down
    2: (0, -1),  # Move left
    3: (0, 1)    # Move right
}

class GridEnv1(gym.Env):
    def __init__(self):
        super(GridEnv1, self).__init__()
        self.action_space = spaces.Discrete(4) # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([ROWS - 1, COLUMNS - 1]), dtype=np.int32)
        self.reward_map = None
        self.obstacle_position = [np.random.randint(ROWS), np.random.randint(COLUMNS)]
        self.step_counter = 0
        self.total_rewards_collected = 0
        self.reset()

    def reset(self):
        self.state = np.array([0, 0]) # Starting at top-left corner
        self.initialize_reward_map()
        self.step_counter = 0  # Reset the step counter at the start of each episode
        self.total_rewards_collected = 0
        return self.state

    def initialize_reward_map(self):
        self.reward_map = np.full((ROWS, COLUMNS), 1)
        self.reward_map[self.obstacle_position[0], self.obstacle_position[1]] = -100
        # self.reward_map[self.state[0], self.state[1]] = 100  # Set the starting column to 0
        # self.reward_map[ROWS-1,0] = 100
        # self.reward_map[-1, :] = 100  # Set the bottom row to 100
            
    def reset_reward_map(self):
        self.initialize_reward_map()
        self.reward_map[0,0] = -1  # Mark the starting position

    def step(self, action):
        # Initialize done as False
        done = False
        reward = 0.
        
        
        # Update state based on action with x, y format
        delta = ACTION_MAPPINGS.get(action, (0, 0))
        new_state = np.array([self.state[0] + delta[0], self.state[1] + delta[1]])
        # if np.array_equal(new_state, prev_state):
        #     reward = -10

        # Check and handle boundary conditions
        if new_state[0] < 0 or new_state[0] >= ROWS or new_state[1] < 0 or new_state[1] >= COLUMNS:
            reward = -100
            done = True
        else:
            # Update the current state and the previous state
            self.state = new_state                  # Update the current state to new state

            self.step_counter += 1
            reward += self.calculate_reward()
            done = self.is_done()

            y = self.state[0]
            x = self.state[1]
            self.reward_map[y, x] = -1

        return self.state, reward, done, {}
            
    def calculate_reward(self):
        # Check if the agent has reached the bottom of the grid
        return self.reward_map[self.state[0], self.state[1]]        

    def is_done(self):
        # Check if all cells in the grid have been visited or if a step limit is reached
        max_steps = MAX_STEPS  # Example step limit
        count_minus_one = np.sum(self.reward_map == -1)
        count_minus_hundred = np.sum(self.reward_map == -100)
        return count_minus_one == (self.reward_map.size - 1) and count_minus_hundred == 1 or self.step_counter >= max_steps    


class QLearningAgent1:
    def __init__(self, env, learning_rate=0.2, discount_factor=0.9, epsilon=0.1, gasoline_capacity=1000, wheat_capacity=100):
        self.env = env
        self.gasoline = gasoline_capacity  # Initial gasoline level
        self.wheat = 0  # Initial wheat level
        self.gasoline_capacity = gasoline_capacity
        self.wheat_capacity = wheat_capacity

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((ROWS, COLUMNS, env.action_space.n))

    def choose_action(self, prev_state, state, neg_reward):

        def get_new_state(action):
            delta = ACTION_MAPPINGS.get(action, (0, 0))
            return np.array([state[0] + delta[0], state[1] + delta[1]])

        def is_valid_action(action):
            new_state = get_new_state(action)
            return not np.array_equal(new_state, prev_state)
        
        def find_second_best_action(q_values):
            # Copy to avoid modifying the original array
            temp_q_values = np.copy(q_values)

            # Find the index of the best action
            best_action = np.argmax(temp_q_values)

            # Mask the best action by setting its value to negative infinity
            temp_q_values[best_action] = -np.inf

            # Find the second best action
            second_best_action = np.argmax(temp_q_values)

            return second_best_action

        action = None
        state_index = (state[0], state[1])
        if neg_reward > 3:
            # Implement logic to find the nearest positive reward
            action = self.find_nearest_positive_reward_action(state)
        elif np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Explore: random action
        else:
            action = np.argmax(self.q_table[state_index])  # Exploit: best known action

        if not is_valid_action(action):
            action = find_second_best_action(self.q_table[state_index])



        # Additional check or fallback strategy if needed
        if not is_valid_action(action):
            # Implement fallback strategy, e.g., select a random action
            action = self.env.action_space.sample()

        return action
    
    def find_nearest_positive_reward_action(self, state):
        min_distance = float('inf')
        best_action = None

        # Search the grid for the nearest positive reward
        for y in range(ROWS):
            for x in range(COLUMNS):
                if self.env.reward_map[y, x] > 0:  # Check for positive reward
                    distance = abs(state[0] - y) + abs(state[1] - x)
                    if distance < min_distance:
                        min_distance = distance
                        best_action = self.determine_action_to_reward(state, (y, x))

        return best_action

    def determine_action_to_reward(self, current_state, reward_state):
        dy = reward_state[0] - current_state[0]
        dx = reward_state[1] - current_state[1]

        if abs(dy) > abs(dx):
            return 1 if dy > 0 else 0  # Move down (1) or up (0) based on the y-difference
        else:
            return 3 if dx > 0 else 2  # Move right (3) or left (2) based on the x-difference
    
    def learn(self, state, action, reward, next_state):
        state_index = (state[0], state[1])
        next_state_index = (next_state[0], next_state[1])
        # Update rule for Q-learning
        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.discount_factor * self.q_table[next_state_index][best_next_action]
        td_error = td_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += self.learning_rate * td_error

class GridEnv2(gym.Env):
    def __init__(self):
        super(GridEnv2, self).__init__()
        self.action_space = spaces.Discrete(4) # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([ROWS - 1, COLUMNS - 1]), dtype=np.int32)
        self.reward_map = None
        self.obstacle_position = [np.random.randint(ROWS), np.random.randint(COLUMNS)]
        self.step_counter = 0
        self.total_rewards_collected = 0
        self.reset()

    def reset(self):
        self.state = np.array([ROWS-1, COLUMNS-1]) # Starting at top-left corner
        self.initialize_reward_map()
        self.step_counter = 0  # Reset the step counter at the start of each episode
        self.total_rewards_collected = 0
        return self.state

    def initialize_reward_map(self):
        self.reward_map = np.full((ROWS, COLUMNS), 1)
        self.reward_map[self.obstacle_position[0], self.obstacle_position[1]] = -100
        # self.reward_map[self.state[0], self.state[1]] = 100  # Set the starting column to 0
        # self.reward_map[ROWS-1,0] = 100
        # self.reward_map[-1, :] = 100  # Set the bottom row to 100
            
    def reset_reward_map(self):
        self.initialize_reward_map()
        self.reward_map[ROWS-1,COLUMNS-1] = -1  # Mark the starting position

    def step(self, action):
        # Initialize done as False
        done = False
        reward = 0.
        
        
        # Update state based on action with x, y format
        delta = ACTION_MAPPINGS.get(action, (0, 0))
        new_state = np.array([self.state[0] + delta[0], self.state[1] + delta[1]])
        # if np.array_equal(new_state, prev_state):
        #     reward = -10

        # Check and handle boundary conditions
        if new_state[0] < 0 or new_state[0] >= ROWS or new_state[1] < 0 or new_state[1] >= COLUMNS:
            reward = -100
            done = True
        else:
            # Update the current state and the previous state
            self.state = new_state                  # Update the current state to new state

            self.step_counter += 1
            reward += self.calculate_reward()
            done = self.is_done()

            y = self.state[0]
            x = self.state[1]
            self.reward_map[y, x] = -1

        return self.state, reward, done, {}
            
    def calculate_reward(self):
        # Check if the agent has reached the bottom of the grid
        return self.reward_map[self.state[0], self.state[1]]        

    def is_done(self):
        # Check if all cells in the grid have been visited or if a step limit is reached
        max_steps = MAX_STEPS  # Example step limit
        count_minus_one = np.sum(self.reward_map == -1)
        count_minus_hundred = np.sum(self.reward_map == -100)
        return count_minus_one == (self.reward_map.size - 1) and count_minus_hundred == 1 or self.step_counter >= max_steps    

class QLearningAgent2:
    def __init__(self, env, learning_rate=0.2, discount_factor=0.9, epsilon=0.1, gasoline_capacity=1000, wheat_capacity=100):
        self.env = env
        self.gasoline = gasoline_capacity  # Initial gasoline level
        self.wheat = 0  # Initial wheat level
        self.gasoline_capacity = gasoline_capacity
        self.wheat_capacity = wheat_capacity

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((ROWS, COLUMNS, env.action_space.n))

    def choose_action(self, prev_state, state, neg_reward):

        def get_new_state(action):
            delta = ACTION_MAPPINGS.get(action, (0, 0))
            return np.array([state[0] + delta[0], state[1] + delta[1]])

        def is_valid_action(action):
            new_state = get_new_state(action)
            return not np.array_equal(new_state, prev_state)
        
        def find_second_best_action(q_values):
            # Copy to avoid modifying the original array
            temp_q_values = np.copy(q_values)

            # Find the index of the best action
            best_action = np.argmax(temp_q_values)

            # Mask the best action by setting its value to negative infinity
            temp_q_values[best_action] = -np.inf

            # Find the second best action
            second_best_action = np.argmax(temp_q_values)

            return second_best_action

        action = None
        state_index = (state[0], state[1])
        if neg_reward > 3:
            # Implement logic to find the nearest positive reward
            action = self.find_nearest_positive_reward_action(state)
        elif np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Explore: random action
        else:
            action = np.argmax(self.q_table[state_index])  # Exploit: best known action

        if not is_valid_action(action):
            action = find_second_best_action(self.q_table[state_index])



        # Additional check or fallback strategy if needed
        if not is_valid_action(action):
            # Implement fallback strategy, e.g., select a random action
            action = self.env.action_space.sample()

        return action
    
    def find_nearest_positive_reward_action(self, state):
        min_distance = float('inf')
        best_action = None

        # Search the grid for the nearest positive reward
        for y in range(ROWS):
            for x in range(COLUMNS):
                if self.env.reward_map[y, x] > 0:  # Check for positive reward
                    distance = abs(state[0] - y) + abs(state[1] - x)
                    if distance < min_distance:
                        min_distance = distance
                        best_action = self.determine_action_to_reward(state, (y, x))

        return best_action

    def determine_action_to_reward(self, current_state, reward_state):
        dy = reward_state[0] - current_state[0]
        dx = reward_state[1] - current_state[1]

        if abs(dy) > abs(dx):
            return 1 if dy > 0 else 0  # Move down (1) or up (0) based on the y-difference
        else:
            return 3 if dx > 0 else 2  # Move right (3) or left (2) based on the x-difference
    
    def learn(self, state, action, reward, next_state):
        state_index = (state[0], state[1])
        next_state_index = (next_state[0], next_state[1])
        # Update rule for Q-learning
        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.discount_factor * self.q_table[next_state_index][best_next_action]
        td_error = td_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += self.learning_rate * td_error


def train_agent(env, agent, episodes):
    best_total_reward = -float('inf')
    best_path = []
    best_wheat_collected = []  # List to keep track of wheat collection

    for episode in range(episodes):
        state = env.reset()
        prev_state = None
        current_path = [state]
        wheat_collected = []  # List for the current episode
        done = False
        total_reward = 0
        neg_reward = 0

        while not done:
            action = agent.choose_action(prev_state, state, neg_reward)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
                
            total_reward += reward
            prev_state = state
            state = next_state
            current_path.append(state)

            if reward == 1:  # Check if wheat is collected
                wheat_collected.append(True)
            else:
                wheat_collected.append(False)

            if reward < 0:
                neg_reward += 1
            else: 
                neg_reward = 0

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_path = current_path
            best_wheat_collected = wheat_collected  # Update the best wheat collection list

    return best_path, best_wheat_collected

# Train the agent
env1 = GridEnv1()
obstacle_position1 = env1.obstacle_position
agent1 = QLearningAgent1(env1)
best_path1, wheat_collected1 = train_agent(env1, agent1, episodes=EPISODES)

env2 = GridEnv2()
obstacle_position2 = env2.obstacle_position
agent2 = QLearningAgent2(env2)
best_path2, wheat_collected2 = train_agent(env2, agent2, episodes=EPISODES)

best_path1_reformat = [element for element in best_path1]
best_path1_modified = [[x[0] + 6, x[1] + 6] for x in best_path1]
best_path2_reformat = [element for element in best_path2]
best_path2_modified = [[x[0] + 6 + 5, x[1] + 6] for x in best_path2]

join_paths = [best_path1_modified, best_path2_modified]

path_with_obstacle = [join_paths, obstacle_position1, obstacle_position2]

def convert_arrays(element):
    if isinstance(element, np.ndarray):
        # Convert NumPy arrays to Python lists
        return element.tolist()
    elif isinstance(element, list):
        # Recursively apply this conversion to each element in the list
        return [convert_arrays(sub_element) for sub_element in element]
    else:
        # If it's neither a NumPy array nor a list, return the element as is
        return element

path_converted = convert_arrays(path_with_obstacle)

# print(join_paths)
print(path_converted)