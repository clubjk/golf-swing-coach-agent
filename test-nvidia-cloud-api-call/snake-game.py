#!/usr/bin/env python3
"""
Simple Snake Game using Pygame.
Controls:
    Arrow keys / WASD – move the snake (first key starts movement)
    P               – pause / resume
    SPACE           – restart after game over
    ESC             – quit the game

Goal: Eat the red food. Each piece makes the snake longer.
If you hit a wall or yourself, it's game over.
"""

import random
import sys
import pygame

# ----------------------------------------------------------------------
# CONSTANTS & COLOURS
# ----------------------------------------------------------------------
CELL_SIZE = 20                # Size of one grid cell in pixels
SCREEN_WIDTH = 640           # Width of the window (must be divisible by CELL_SIZE)
SCREEN_HEIGHT = 480          # Height of the window
FPS_START = 10               # Starting frames per second (game speed)
FPS_MAX = 30                 # Max speed (prevents it from becoming unplayable)

BG_COLOR = (20, 20, 20)      # Dark background
GRID_COLOR = (40, 40, 40)    # Subtle grid lines (optional)
SNAKE_COLOR = (0, 255, 0)    # Bright green snake
SNAKE_HEAD_COLOR = (0, 200, 0)  # Slightly darker head for emphasis
FOOD_COLOR = (255, 0, 0)     # Red food
TEXT_COLOR = (255, 255, 255) # White UI text
PAUSE_COLOR = (200, 200, 0)  # Yellow text while paused

DIRECTION_KEYS = frozenset(
    (
        pygame.K_UP,
        pygame.K_DOWN,
        pygame.K_LEFT,
        pygame.K_RIGHT,
        pygame.K_w,
        pygame.K_s,
        pygame.K_a,
        pygame.K_d,
    )
)

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def get_grid_rect(x, y):
    """Return a pygame.Rect for a grid cell (x,y)."""
    return pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

def random_food_position(snake):
    """Return a random (x, y) that is not occupied by the snake."""
    cols = SCREEN_WIDTH // CELL_SIZE
    rows = SCREEN_HEIGHT // CELL_SIZE
    while True:
        pos = (random.randint(0, cols - 1), random.randint(0, rows - 1))
        if pos not in snake:
            return pos

def direction_from_key(key, current_dir, snake_length):
    """
    Return a new direction tuple (dx, dy) based on key press.
    With only a head (length 1), any direction is fine — there is no body to
    reverse into. With 2+ segments, disallow instant 180° turns.
    """
    mapping = {
        pygame.K_UP:    (0, -1),
        pygame.K_DOWN:  (0, 1),
        pygame.K_LEFT:  (-1, 0),
        pygame.K_RIGHT: (1, 0),
        pygame.K_w:     (0, -1),
        pygame.K_s:     (0, 1),
        pygame.K_a:     (-1, 0),
        pygame.K_d:     (1, 0),
    }
    if key not in mapping:
        return current_dir
    new_dir = mapping[key]
    if snake_length > 1 and new_dir[0] == -current_dir[0] and new_dir[1] == -current_dir[1]:
        return current_dir
    return new_dir

# ----------------------------------------------------------------------
# GAME CLASS
# ----------------------------------------------------------------------
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🐍 Snake Game – Have fun!")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.large_font = pygame.font.SysFont("monospace", 48, bold=True)

        self.reset_game()
        self.main_loop()

    def reset_game(self):
        """Initialize / reset all game state variables."""
        # Snake starts in the middle, length 1 (no move until first arrow/WASD)
        start_x = (SCREEN_WIDTH // CELL_SIZE) // 2
        start_y = (SCREEN_HEIGHT // CELL_SIZE) // 2
        self.snake = [(start_x, start_y)]
        self.direction = (1, 0)          # first keypress replaces this
        self.waiting_for_first_move = True
        self.food = random_food_position(self.snake)
        self.score = 0
        self.fps = FPS_START
        self.game_over = False
        self.paused = False

    # ------------------------------------------------------------------
    # GAME LOGIC
    # ------------------------------------------------------------------
    def move_snake(self):
        """Move the snake one cell forward, check for collisions."""
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Wall collision
        cols = SCREEN_WIDTH // CELL_SIZE
        rows = SCREEN_HEIGHT // CELL_SIZE
        if not (0 <= new_head[0] < cols and 0 <= new_head[1] < rows):
            self.game_over = True
            return

        # Self collision (ignore the last tail piece because it will be removed)
        if new_head in self.snake[:-1]:
            self.game_over = True
            return

        # Add new head
        self.snake.insert(0, new_head)

        # Food?
        if new_head == self.food:
            self.score += 1
            self.food = random_food_position(self.snake)
            # Speed up a little (capped at FPS_MAX)
            self.fps = min(self.fps + 1, FPS_MAX)
        else:
            # Remove tail if we didn't eat
            self.snake.pop()

    # ------------------------------------------------------------------
    # RENDERING
    # ------------------------------------------------------------------
    def draw_background(self):
        self.screen.fill(BG_COLOR)
        # Optional subtle grid lines (makes movement easier to see)
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y))

    def draw_food(self):
        rect = get_grid_rect(*self.food)
        # Slightly smaller rectangle for a nicer look
        inset = 2
        pygame.draw.rect(self.screen, FOOD_COLOR,
                         rect.inflate(-inset * 2, -inset * 2))

    def draw_snake(self):
        for i, (x, y) in enumerate(self.snake):
            rect = get_grid_rect(x, y)
            color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_COLOR
            pygame.draw.rect(self.screen, color, rect)
            # Small border for each segment
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

    def draw_score(self):
        label = self.font.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.screen.blit(label, (10, 10))

    def draw_pause(self):
        txt = self.large_font.render("PAUSED", True, PAUSE_COLOR)
        rect = txt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        # Semi‑transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(txt, rect)

    def draw_game_over(self):
        txt1 = self.large_font.render("GAME OVER", True, (255, 50, 50))
        txt2 = self.font.render(f"Final Score: {self.score}", True, TEXT_COLOR)
        txt3 = self.font.render("Press SPACE to restart", True, TEXT_COLOR)
        rect1 = txt1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40))
        rect2 = txt2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 10))
        rect3 = txt3.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        # Dark overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(txt1, rect1)
        self.screen.blit(txt2, rect2)
        self.screen.blit(txt3, rect3)

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def main_loop(self):
        while True:
            # -------- Event handling ---------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    # Global controls
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_p:
                        self.paused = not self.paused

                    if self.game_over:
                        if event.key == pygame.K_SPACE:
                            self.reset_game()
                        continue

                    # Direction change (only when not paused)
                    if not self.paused and event.key in DIRECTION_KEYS:
                        self.direction = direction_from_key(
                            event.key, self.direction, len(self.snake)
                        )
                        self.waiting_for_first_move = False

            # -------- Update -----------------------------------------
            if (
                not self.paused
                and not self.game_over
                and not self.waiting_for_first_move
            ):
                self.move_snake()

            # -------- Render -----------------------------------------
            self.draw_background()
            self.draw_food()
            self.draw_snake()
            self.draw_score()

            if self.paused and not self.game_over:
                self.draw_pause()
            if self.game_over:
                self.draw_game_over()

            pygame.display.flip()
            self.clock.tick(self.fps)

# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SnakeGame()