import sys; sys.path.insert(0, 'model')
from predict import _build_descriptive_fallback

functions = {
    'print_board': ('def print_board(board):\n    for row in board:\n        print(" | ".join(row))\n        print("-" * 9)', 'function'),
    'check_winner': ('def check_winner(board, player):\n    for row in board:\n        if all([cell == player for cell in row]):\n            return True\n    for col in range(3):\n        if all([board[row][col] == player for row in range(3)]):\n            return True\n    if all([board[i][i] == player for i in range(3)]):\n        return True\n    if all([board[i][2 - i] == player for i in range(3)]):\n        return True\n    return False', 'function'),
    'is_full': ('def is_full(board):\n    return all([cell != " " for row in board for cell in row])', 'function'),
    'main': ('def main():\n    board = [["X"] for _ in range(3)]\n    current_player = "X"\n    while True:\n        move = input("Player")', 'function'),
    'loop_for_row': ('for row in board:\n    print(" | ".join(row))\n    print("-" * 9)', 'loop'),
    'loop_for_board_check': ('for row in board:\n    if all([cell == player for cell in row]):\n        return True', 'loop'),
    'loop_for_range': ('for col in range(3):\n    if all([board[row][col] == player for row in range(3)]):\n        return True', 'loop'),
    'loop_while': ('while True:\n    try:\n        move = input("Player")\n    except:\n        pass', 'loop'),
    'var_board': ('board = [[" " for _ in range(3)] for _ in range(3)]', 'variable'),
    'var_move': ('move = input(f"Player {current_player}")', 'variable'),
    'logic_try': ('try:\n    move = input("Player")\n    row, col = map(int, move.split())\n    if not (0 <= row <= 2):\n        print("Invalid")\n        continue\nexcept (ValueError, IndexError):\n    print("Invalid input!")', 'complex_logic'),
}

for label, (code, ctype) in functions.items():
    comment, rule = _build_descriptive_fallback(code, ctype)
    print(f'{label:25s} | {comment}')
