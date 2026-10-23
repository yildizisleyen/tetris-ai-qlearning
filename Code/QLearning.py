import matplotlib.pyplot as plt
import numpy as np
import random
import time


class TetrisQLearning:
    def __init__(self, rows=20, cols=10, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.q_table = {}       # Q-Table for state-action pairs
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.score = 0
        self.current_piece = self.get_random_piece()

    def get_random_piece(self):
        pieces = [
            (np.array([[1, 1, 1, 1]]), 1), 
            (np.array([[1, 1], [1, 1]]), 2),  
            (np.array([[0, 1, 1], [1, 1, 0]]), 3),  
            (np.array([[1, 1, 0], [0, 1, 1]]), 4),  
            (np.array([[1, 1, 1], [0, 1, 0]]), 5), 
        ]
        return random.choice(pieces)

    def render_board(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
            ax.set_xticks([])  
            ax.set_yticks([])  
            plt.ion()  
            fig.show()

        if not plt.fignum_exists(fig.number):
            print("visualization window was closed, stopping the process.")
            return None, None

        ax.clear()

        colors = [
            (1, 1, 1),
            (0.0, 1.0, 1.0), 
            (1.0, 1.0, 0.0), 
            (0.0, 1.0, 0.0),  
            (1.0, 0.0, 0.0),  
            (1.0, 0.0, 1.0),  
        ]
        cmap = plt.cm.colors.ListedColormap(colors)

        ax.imshow(self.board, cmap=cmap, origin="upper")
        ax.set_title(f"Score: {self.score}")
        fig.canvas.draw()  
        fig.canvas.flush_events()  
        return fig, ax

    def get_state(self):
        return tuple(self.board.flatten())

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.get_possible_actions())
        q_values = self.q_table.get(state, {})
        return max(q_values, key=q_values.get, default=random.choice(self.get_possible_actions()))

    def get_possible_actions(self):
        actions = []
        for rotation in range(4):
            for col in range(self.cols):
                actions.append((rotation, col))
        return actions

    def take_action(self, action):
        rotation, col = action
        piece, piece_color = self.rotate_piece(self.current_piece[0], rotation), self.current_piece[1]
        if not self.is_valid_position(piece, col):
            return -1  # punishment

        self.place_piece(piece, col, piece_color)
        lines_cleared = self.clear_lines()
        self.score += lines_cleared
        self.current_piece = self.get_random_piece()
        return lines_cleared

    def rotate_piece(self, piece, rotation):
        piece = np.array(piece) 
        for _ in range(rotation):
            piece = np.rot90(piece)
        return piece

    def is_valid_position(self, piece, col):
        piece_width = piece.shape[1]
        if col + piece_width > self.cols:
            return False
        return True

    def place_piece(self, piece, col, piece_color):
        for row in range(self.rows - piece.shape[0], -1, -1):
            if self.can_place(piece, row, col):
                for r in range(piece.shape[0]):
                    for c in range(piece.shape[1]):
                        if piece[r, c] == 1:  
                            self.board[row + r, col + c] = piece_color  
                break

    def can_place(self, piece, row, col):
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1: 
                    if row + r >= self.rows or col + c >= self.cols or self.board[row + r, col + c] != 0:
                        return False  
        return True

    def clear_lines(self):
        full_lines = [i for i in range(self.rows) if all(self.board[i])]
        for line in full_lines:
            self.board[1:line + 1] = self.board[:line]
            self.board[0] = 0
        return len(full_lines) * 10

    def update_q_table(self, state, action, reward, next_state):
        q_values = self.q_table.setdefault(state, {})
        next_q_values = self.q_table.get(next_state, {})
        max_next_q = max(next_q_values.values(), default=0)
        q_values[action] = q_values.get(action, 0) + self.alpha * (reward + self.gamma * max_next_q - q_values.get(action, 0))

    def train(self, episodes=1000):
        fig, ax = None, None
        for episode in range(episodes):
            self.reset()
            state = self.get_state()
            done = False
            step = 0
            max_steps = 500

            start_time = time.time() 


            while not done and step < max_steps:
                fig, ax = self.render_board(fig, ax)
                if fig is None or ax is None:
                    print("Visualization window was closed, stopping the process.")
                    return
                action = self.choose_action(state)
                reward = self.take_action(action)
                next_state = self.get_state()
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                done = self.is_game_over()
                step += 1

            end_time = time.time()  
            game_duration = end_time - start_time

            print(f"Episode {episode + 1} duration: {game_duration:.2f} seconds")
            print(f"Episode {episode + 1} score: {self.score}")


        plt.ioff() 
        plt.show()  


    def is_game_over(self):
        return any(self.board[0])
    
    def choose_greedy_action(self, state):
        q_values = self.q_table.get(state, {})
        if q_values:
            return max(q_values, key=q_values.get) 
        else:
            return random.choice(self.get_possible_actions())  


if __name__ == "__main__":
    agent = TetrisQLearning()
    agent.train(episodes=500)

def test_model(agent, episodes=10):
    total_score = 0
    for episode in range(episodes):
        agent.reset()
        state = agent.get_state()
        done = False
        fig, ax = None, None  
        while not done:
            fig, ax = agent.render_board(fig, ax) 
            action = agent.choose_action(state) 
            agent.take_action(action)
            state = agent.get_state()
            done = agent.is_game_over()
        total_score += agent.score
        print(f"Episode {episode + 1}: Score = {agent.score}")
    print(f"Average Score over {episodes} episodes: {total_score / episodes}")

def test_greedy_model(agent, episodes=10):
    total_score = 0
    for episode in range(episodes):
        agent.reset()
        state = agent.get_state()
        done = False
        fig, ax = None, None  
        while not done:
            fig, ax = agent.render_board(fig, ax)  
            action = agent.choose_greedy_action(state) 
            agent.take_action(action)
            state = agent.get_state()
            done = agent.is_game_over()
        total_score += agent.score
        print(f"Episode {episode + 1}: Score = {agent.score}")
    print(f"Average Score over {episodes} episodes: {total_score / episodes}")


if __name__ == "__main__":
    agent_qlearning = TetrisQLearning()
    print("Q-learning model is testing...")
    test_model(agent_qlearning, episodes=2)

    agent_greedy = TetrisQLearning()
    print("Greedy model is testing...")
    test_greedy_model(agent_greedy, episodes=2)

print(f"Q-learning epsilon: {agent.epsilon}")
