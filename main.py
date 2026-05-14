import random

import numpy as np

T = 15000

ROWS = 5
COLS = 5
CONNECT = 4

WIN_REWARD = 100
LOSS_PENALTY = -100
STEP_PENALTY = -1
DRAW_REWARD = 20


class ConnectFour:
    def __init__(self):
        # 0 = empty, 1 = player 1, 2 = player 2
        self.board = np.zeros((ROWS, COLS), dtype=int)

    # restituisce la griglia come tupla di tuple
    def get_state(self):
        state = []
        for row in self.board:
            state.append(tuple(row))
        return tuple(state)

    # colonne giocabili
    def available_actions(self):
        available = []
        for c in range(COLS):
            if self.board[0][c] == 0:
                available.append(c)
        return available

    # inserisce il disco
    def drop_piece(self, column, player):
        for r in reversed(range(ROWS)):
            if self.board[r][column] == 0:
                self.board[r][column] = player
                return

    # controlla se ha vinto
    def check_win(self, player):
        for r in range(ROWS):
            for c in range(COLS - CONNECT + 1):
                if all(self.board[r, c + i] == player for i in range(CONNECT)):
                    return True
        for r in range(ROWS - CONNECT + 1):
            for c in range(COLS):
                if all(self.board[r + i, c] == player for i in range(CONNECT)):
                    return True
        for r in range(ROWS - CONNECT + 1):
            for c in range(COLS - CONNECT + 1):
                if all(self.board[r + i, c + i] == player for i in range(CONNECT)):
                    return True
                if all(
                    self.board[r + CONNECT - 1 - i, c + i] == player
                    for i in range(CONNECT)
                ):
                    return True
        return False

    # esegue una mossa
    def step(self, column, player):
        self.drop_piece(column, player)

        if self.check_win(player):
            return self.get_state(), WIN_REWARD, True

        # nessuna colonna disponibile: pareggio
        if len(self.available_actions()) == 0:
            return self.get_state(), DRAW_REWARD, True

        return self.get_state(), STEP_PENALTY, False

    # stampa griglia attuale
    def print_board(self):
        print(" ".join(str(c) for c in range(COLS)))
        for row in self.board:
            line = ""
            for cell in row:
                if cell == 0:
                    line += "."
                elif cell == 1:
                    line += "X"
                else:
                    line += "O"
                line += " "
            print(line)
        print()


# Qlearning
class QLearningAgent:
    def __init__(self):
        # dizionario (stato griglia, azione) -> q
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05

    def get_q(self, state, action):
        key = (state, action)
        if key not in self.q_table:
            return 0.0
        return self.q_table[key]

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)

        q_values = []
        for a in actions:
            q_values.append(self.get_q(state, a))

        best_q = max(q_values)

        best_actions = []
        for i in range(len(actions)):
            if q_values[i] == best_q:
                best_actions.append(actions[i])

        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_actions):
        current_q = self.get_q(state, action)

        if len(next_actions) == 0:
            best_future_q = 0
        else:
            future_qs = []
            for a in next_actions:
                future_qs.append(self.get_q(next_state, a))
            best_future_q = max(future_qs)

        target = reward + self.discount * best_future_q
        new_q = current_q + self.learning_rate * (target - current_q)

        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min


# reinforcement learnign
env = ConnectFour()
agent1 = QLearningAgent()  # X
agent2 = QLearningAgent()  # O

# 1 = agent1 vince, 2 = agent2 vince, 0 = patta
results = []

for episode in range(T):
    env.board[:, :] = 0
    state = env.get_state()
    game_over = False

    while not game_over:
        # agent1
        actions = env.available_actions()
        action1 = agent1.choose_action(state, actions)
        state_after_1, reward1, game_over = env.step(action1, 1)

        if game_over:
            agent1.learn(state, action1, reward1, state_after_1, [])
            if reward1 == WIN_REWARD:
                results.append(1)
            else:
                results.append(0)
            break

        # agent2
        actions2 = env.available_actions()
        action2 = agent2.choose_action(state_after_1, actions2)
        state_after_2, reward2, game_over = env.step(action2, 2)

        if game_over:
            agent2.learn(state_after_1, action2, reward2, state_after_2, [])
            if reward2 == WIN_REWARD:
                reward1 = LOSS_PENALTY
                results.append(2)
            else:
                reward1 = DRAW_REWARD
                results.append(0)
            agent1.learn(state, action1, reward1, state_after_2, [])
            break

        next_actions = env.available_actions()
        agent1.learn(state, action1, STEP_PENALTY, state_after_2, next_actions)
        agent2.learn(state_after_1, action2, STEP_PENALTY, state_after_2, next_actions)

        state = state_after_2

    agent1.decay_epsilon()
    agent2.decay_epsilon()

    if (episode + 1) % 500 == 0:
        # risultati delle ultime 500 partite
        recent = results[-500:]
        a1 = recent.count(1) / len(recent)
        a2 = recent.count(2) / len(recent)
        draw = max(0.0, 1 - a1 - a2)
        print(
            f"episode {episode + 1}: agent1 {a1:.0%}  agent2 {a2:.0%} draw {draw:.0%} epsilon {agent1.epsilon:.2f}"
        )

total = len(results)
a1 = results.count(1) / total
a2 = results.count(2) / total
print(f"result: agent1 {a1:.1%}  agent2 {a2:.1%}  draw {1 - a1 - a2:.1%}")

# gioca contro l'agente migliore
if a1 >= a2:
    best = agent1
    best_player = 1
else:
    best = agent2
    best_player = 2

# disattiva epsilon per giocare
best.epsilon = 0
your_player = 3 - best_player

print()
print(f"Gioca contro l'agente migliore")

env.board[:, :] = 0
state = env.get_state()
game_over = False
env.print_board()

turn = 1
while not game_over:
    if turn == your_player:
        while True:
            try:
                col = int(input(f"inserisci num colonna (0-{COLS - 1}): "))
                if col in env.available_actions():
                    break
                print("column full or invalid, try again")
            except ValueError:
                print(f"inserisci num colonna (0-{COLS - 1}): ")

        state, reward, game_over = env.step(col, your_player)
        env.print_board()

        if game_over:
            if reward == WIN_REWARD:
                print("YOU WIN!")
            else:
                print("DRAW")
            break

    else:
        actions = env.available_actions()
        action = best.choose_action(state, actions)
        print(f"agente ha giocato nella colonna {action}")
        state, reward, game_over = env.step(action, best_player)
        env.print_board()

        if game_over:
            if reward == WIN_REWARD:
                print("AGENT WINS!")
            else:
                print("DRAW")
            break

    turn = 3 - turn  # scambia tra 1 e 2
