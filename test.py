import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class MaizField:
    def _init_(self, size=10):
        self.original_size = size  # Tamaño original
        self.size = size * 2  # Duplicar el tamaño
        self.field = np.zeros((self.size, self.size))  # Inicializar con ceros
        self.tractores_positions = [(0, 0), (0, self.size - 1)]  # Posiciones iniciales de los tractores
        self.place_original_maiz()  # Colocar maíz en la parte original
        self.total_maiz = np.sum(self.field)

    def place_original_maiz(self):
        for x in range(self.original_size):
            for y in range(self.original_size):
                self.field[x, y] = 1

    def reset(self):
        self.field = np.zeros((self.size, self.size))  # Resetear con ceros
        self.place_original_maiz()  # Colocar el maíz nuevamente
        self.tractores_positions = [(0, 0), (0, self.size - 1)]
        self.total_maiz = np.sum(self.field)
        return [self.state_to_index(pos) for pos in self.tractores_positions]

    def state_to_index(self, state):
        return state[0] * self.size + state[1]

    def index_to_state(self, index):
        return (index // self.size, index % self.size)

    def step(self, actions):
        rewards = []
        new_positions = []
        for i, action in enumerate(actions):
            x, y = self.tractores_positions[i]
            if action == 0 and x > 0: x -= 1  # Arriba
            elif action == 1 and x < self.size - 1: x += 1  # Abajo
            elif action == 2 and y > 0: y -= 1  # Izquierda
            elif action == 3 and y < self.size - 1: y += 1  # Derecha

            new_positions.append((x, y))

        # Verificar colisiones
        if new_positions[0] == new_positions[1]:
            return [self.state_to_index(pos) for pos in self.tractores_positions], [-1, -1]  # Penalización por colisión

        # Actualizar posiciones y recompensas
        for i, pos in enumerate(new_positions):
            self.tractores_positions[i] = pos
            reward = 1 if self.field[pos] == 1 else -1
            self.field[pos] = 0  # Recolectar el maíz
            self.total_maiz -= 1
            rewards.append(reward)

        return [self.state_to_index(pos) for pos in self.tractores_positions], rewards

class QLearningAgent:
    def _init_(self, states, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = np.zeros((states, actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice([0, 1, 2, 3])  # Explorar
        else:
            return np.argmax(self.q_table[state])  # Explotar

    def update_q_table(self, state, action, reward, new_state):
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[new_state])
        self.q_table[state, action] += self.learning_rate * (target - predict)

def train_and_visualize(episodes=5, size=10, steps_per_episode=100):
    field = MaizField(size=size)
    agent1 = QLearningAgent(size**2 * 4, 4)  # Ajustar el número de estados
    agent2 = QLearningAgent(size**2 * 4, 4)  # Segundo agente

    fig, ax = plt.subplots()
    cmap = ListedColormap(['white', 'yellow', 'red', 'blue'])  # Colores para la visualización

    for episode in range(episodes):
        states = field.reset()
        ax.clear()
        ax.set_title(f"Episode: {episode + 1}")
        for step in range(steps_per_episode):
            action1 = agent1.choose_action(states[0])
            action2 = agent2.choose_action(states[1])
            new_states, rewards = field.step([action1, action2])
            agent1.update_q_table(states[0], action1, rewards[0], new_states[0])
            agent2.update_q_table(states[1], action2, rewards[1], new_states[1])
            states = new_states

            # Actualizar la visualización
            data = np.copy(field.field)
            for i, pos in enumerate(field.tractores_positions):
                data[pos] = 2 + i  # Diferenciar los tractores
            plot = ax.imshow(data.T, cmap=cmap, interpolation='nearest')
            ax.set_title(f"Episode: {episode + 1}, Step: {step + 1}")
            plt.pause(0.05)

            if field.total_maiz == 0:
                break

        plt.pause(1)  # Pausa entre episodios

    plt.show()

# Ejecutar la simulación y visualización
train_and_visualize()