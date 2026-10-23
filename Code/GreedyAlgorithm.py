import numpy as np
import random
import pygame

BOARD_HEIGHT = 20
BOARD_WIDTH = 10
CELL_SIZE = 30  

tetrominoes = {
    'I': np.array([[1, 1, 1, 1]]),  
    'O': np.array([[1, 1], [1, 1]]),  
    'T': np.array([[0, 1, 0], [1, 1, 1]]),  
    'L': np.array([[1, 0], [1, 0], [1, 1]]),  
    'J': np.array([[0, 1], [0, 1], [1, 1]]),  
    'S': np.array([[0, 1, 1], [1, 1, 0]]),  
    'Z': np.array([[1, 1, 0], [0, 1, 1]]),  
}

board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

pygame.init()
screen = pygame.display.set_mode((BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Tetris AI Visualization")

BACKGROUND_COLOR = (0, 0, 0)  
GRID_COLOR = (50, 50, 50)    
BLOCK_COLOR = (0, 255, 0)     

def draw_board(board):
    screen.fill(BACKGROUND_COLOR)

    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            if board[row][col] == 1:
                pygame.draw.rect(
                    screen,
                    BLOCK_COLOR,
                    (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )
            pygame.draw.rect(
                screen,
                GRID_COLOR,
                (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                1  
            )

    pygame.display.update()


def place_tetromino(board, tetromino_number, x, y):
    tetromino = tetrominoes[tetromino_number]

    for i in range(tetromino.shape[0]):  
        for j in range(tetromino.shape[1]):  
            if tetromino[i][j] == 1: 
                board_y = y + i
                board_x = x + j

                if board_y >= BOARD_HEIGHT or board_x >= BOARD_WIDTH or board_x < 0:
                    return board  

                if board[board_y][board_x] == 1:
                    return board 

                if board_y < BOARD_HEIGHT - 1 and tetromino[i][j] == 1:
                    if board[board_y + 1][board_x] == 0 and (i == tetromino.shape[0] - 1):
                        return board 

    for i in range(tetromino.shape[0]):
        for j in range(tetromino.shape[1]):
            if tetromino[i][j] == 1:
                board[y + i][x + j] = 1  
    return board


def clear_lines(board):
    new_board = np.copy(board)
    full_lines = []
    
    for i in range(BOARD_HEIGHT):
        if np.all(new_board[i] == 1): 
            full_lines.append(i)
    
    for line in full_lines:
        new_board = np.delete(new_board, line, 0)  
        new_board = np.vstack([np.zeros((1, BOARD_WIDTH)), new_board]) 
    
    return new_board

def evaluate_board(board):
    lines_cleared = 0
    max_height = 0
    holes = 0
    bumpiness = 0
    board = clear_lines(board)
    for row in range(BOARD_HEIGHT):
        if np.all(board[row] == 1): 
            lines_cleared += 1
        for col in range(BOARD_WIDTH):
            if board[row][col] == 0:  
                holes += 1
            max_height = max(max_height, row)

    for col in range(1, BOARD_WIDTH):
        bumpiness += abs(np.sum(board[:, col]) - np.sum(board[:, col - 1]))

    return lines_cleared, max_height, holes, bumpiness

def greedy_move(board, tetromino_number):
    best_move = None
    best_score = float('inf')  

    tetromino = tetrominoes[tetromino_number]
    for rotation in range(4):  
        rotated_piece = np.rot90(tetromino, rotation)
        for x in range(BOARD_WIDTH - rotated_piece.shape[1] + 1):
            for y in range(BOARD_HEIGHT - rotated_piece.shape[0], -1, -1):
                temp_board = board.copy()  
                
                is_valid = True
                for i in range(rotated_piece.shape[0]):
                    for j in range(rotated_piece.shape[1]):
                        if rotated_piece[i][j] == 1:
                            if y + i >= BOARD_HEIGHT or x + j >= BOARD_WIDTH or x + j < 0:
                                is_valid = False
                                break
                            if board[y + i][x + j] != 0:  
                                is_valid = False
                                break
                if not is_valid:
                    continue

                temp_board = place_tetromino(temp_board, tetromino_number, x, y)

                lines_cleared, max_height, holes, bumpiness = evaluate_board(temp_board)
                score = max_height + holes + bumpiness  

                if score < best_score:
                    best_score = score
                    best_move = (x, y, rotated_piece, score) 

    return best_move

pygame.font.init()
font = pygame.font.Font(None, 36)  

def draw_score(score):
    score_text = f"Score: {score}"
    
    score_surface = font.render(score_text, True, (255, 255, 255))  
    
    screen.blit(score_surface, (10, 10))  

def play_tetris(board, max_moves=10):
    total_score = 0  
    for move_count in range(1, max_moves + 1): 
        tetromino_number = random.choice(list(tetrominoes.keys())) 
        move = greedy_move(board, tetromino_number)

        if move:
            x, y, piece, score = move
            board = place_tetromino(board, tetromino_number, x, y) 
            total_score += score 
            
            draw_board(board)
            draw_score(total_score) 
            pygame.display.update()  
            
            pygame.time.delay(500) 
        else:
            print("Game Over!")
            break  

play_tetris(board, max_moves=10)
pygame.quit()



