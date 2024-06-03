import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tic-Tac-Toe")

        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.human_player = 'O'
        self.ai_player = 'X'
        self.current_player = self.human_player
        self.rounds = 0
        self.human_score = 0
        self.ai_score = 0

        self.buttons = [[None for _ in range(3)] for _ in range(3)]

        self.create_widgets()

        self.window.mainloop()

    def create_widgets(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(self.window, text=' ', font='Arial 20', width=5, height=2,
                                               command=lambda i=i, j=j: self.on_button_click(i, j))
                self.buttons[i][j].grid(row=i, column=j)

        self.label_rounds = tk.Label(self.window, text=f"Rounds: {self.rounds}", font='Arial 14')
        self.label_rounds.grid(row=3, column=0, columnspan=3)

        self.label_scores = tk.Label(self.window, text=f"Scores - Player: {self.human_score}, AI: {self.ai_score}", font='Arial 14')
        self.label_scores.grid(row=4, column=0, columnspan=3)

        self.reset_button = tk.Button(self.window, text='Reset', font='Arial 14', command=self.reset_game)
        self.reset_button.grid(row=5, column=0, columnspan=3)

    def on_button_click(self, row, col):
        if self.board[row][col] == ' ' and self.current_player == self.human_player:
            self.make_move(row, col, self.human_player)
            self.current_player = self.ai_player
            self.update_buttons()
            if self.check_winner() or self.is_board_full():
                self.end_round()
            else:
                self.ai_move()

    def ai_move(self):
        move = self.find_best_move_ab(self.board)
        if move:
            self.make_move(move[0], move[1], self.ai_player)
            self.current_player = self.human_player
            self.update_buttons()
            if self.check_winner() or self.is_board_full():
                self.end_round()

    def make_move(self, row, col, player):
        self.board[row][col] = player

    def undo_move(self, row, col):
        self.board[row][col] = ' '

    def update_buttons(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text=self.board[i][j])

    def check_winner(self):
        for row in self.board:
            if row[0] == row[1] == row[2] and row[0] != ' ':
                return row[0]
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col] != ' ':
                return self.board[0][col]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != ' ':
            return self.board[0][2]
        return None

    def is_board_full(self):
        for row in self.board:
            if ' ' in row:
                return False
        return True

    def end_round(self):
        winner = self.check_winner()
        if winner == self.human_player:
            self.human_score += 1
            messagebox.showinfo("Round Over", "You win!")
        elif winner == self.ai_player:
            self.ai_score += 1
            messagebox.showinfo("Round Over", "AI wins!")
        else:
            messagebox.showinfo("Round Over", "It's a draw!")

        self.rounds += 1
        self.reset_board()
        self.update_labels()

    def reset_board(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = self.human_player
        self.update_buttons()

    def reset_game(self):
        self.reset_board()
        self.rounds = 0
        self.human_score = 0
        self.ai_score = 0
        self.update_labels()

    def update_labels(self):
        self.label_rounds.config(text=f"Rounds: {self.rounds}")
        self.label_scores.config(text=f"Scores - Player: {self.human_score}, AI: {self.ai_score}")

    def minimax_ab(self, board, depth, alpha, beta, is_maximizing):
        score = self.evaluate(board)

        if score == 1 or score == -1:
            return score

        if self.is_board_full():
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for (i, j) in self.get_empty_cells(board):
                self.make_move(i, j, self.ai_player)
                score = self.minimax_ab(board, depth + 1, alpha, beta, False)
                self.undo_move(i, j)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for (i, j) in self.get_empty_cells(board):
                self.make_move(i, j, self.human_player)
                score = self.minimax_ab(board, depth + 1, alpha, beta, True)
                self.undo_move(i, j)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score

    def find_best_move_ab(self, board):
        best_score = float('-inf')
        best_move = None

        for (i, j) in self.get_empty_cells(board):
            self.make_move(i, j, self.ai_player)
            move_score = self.minimax_ab(board, 0, float('-inf'), float('inf'), False)
            self.undo_move(i, j)

            if move_score > best_score:
                best_score = move_score
                best_move = (i, j)

        return best_move

    def get_empty_cells(self, board):
        cells = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    cells.append((i, j))
        return cells

    def evaluate(self, board):
        winner = self.check_winner()
        if winner == self.ai_player:
            return 1
        elif winner == self.human_player:
            return -1
        else:
            return 0

if __name__ == "__main__":
    TicTacToe()
