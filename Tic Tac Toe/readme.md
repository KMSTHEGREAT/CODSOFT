# Tic-Tac-Toe Game with AI

This project implements a Tic-Tac-Toe game with a graphical user interface (GUI) using Tkinter in Python. The game allows a human player to play against an AI opponent. The AI uses the minimax algorithm with alpha-beta pruning to determine the best moves.

## Features

- GUI built with Tkinter.
- Human player vs AI player.
- AI uses the minimax algorithm with alpha-beta pruning.
- Score tracking for both players.
- Round tracking and display.
- Reset button to restart the game.

## Requirements

- Python 3.x
- Tkinter (usually comes pre-installed with Python)

## How to Run

1. Ensure you have Python 3 installed on your system.
2. Clone this repository or download the source code.
3. Navigate to the directory containing `main.py`.
4. Run the game with the following command:
   ```sh
   python main.py
   ```

## Files

- `main.py`: Contains the implementation of the Tic-Tac-Toe game, including the GUI and AI logic.

## Classes and Methods

### TicTacToe Class

- `__init__(self)`: Initializes the game, sets up the board and GUI.
- `create_widgets(self)`: Creates the buttons, labels, and reset button for the GUI.
- `on_button_click(self, row, col)`: Handles the event when a button is clicked by the human player.
- `ai_move(self)`: Executes the AI player's move.
- `make_move(self, row, col, player)`: Makes a move on the board.
- `undo_move(self, row, col)`: Undoes a move on the board.
- `update_buttons(self)`: Updates the text on the buttons to reflect the board state.
- `check_winner(self)`: Checks if there is a winner.
- `is_board_full(self)`: Checks if the board is full.
- `end_round(self)`: Ends the current round, updates scores, and resets the board.
- `reset_board(self)`: Resets the board for a new round.
- `reset_game(self)`: Resets the entire game, including scores and rounds.
- `update_labels(self)`: Updates the labels displaying rounds and scores.
- `minimax_ab(self, board, depth, alpha, beta, is_maximizing)`: Implements the minimax algorithm with alpha-beta pruning.
- `find_best_move_ab(self, board)`: Finds the best move for the AI player using the minimax algorithm.
- `get_empty_cells(self, board)`: Returns a list of empty cells on the board.
- `evaluate(self, board)`: Evaluates the board to determine the score.

## Gameplay

- The human player always starts as 'O'.
- The AI player is 'X'.
- The game continues until there is a winner or the board is full.
- The scores and rounds are displayed at the bottom of the window.
- The "Reset" button can be used to restart the game, resetting scores and rounds.

## Future Improvements

- Add difficulty levels by adjusting the depth of the minimax algorithm.
- Add a main menu to choose between single-player and two-player modes.
- Improve the GUI with better graphics and animations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The minimax algorithm with alpha-beta pruning is based on standard implementations used in game AI.

Enjoy playing Tic-Tac-Toe!
This `README.md` file provides a clear and comprehensive overview of the project, instructions on how to run it, and details about the implemented classes and methods.