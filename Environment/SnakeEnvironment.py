import random
import sys
from threading import Thread
import pygame
import time

class SnakeEnvironment:
    def __init__(self, x_blocks, y_blocks, block_size):
        self.block_size = block_size
        self.x_blocks = x_blocks
        self.y_blocks = y_blocks
        pygame.init()
        # create the display surface object of specific dimension.
        self.window = pygame.display.set_mode((x_blocks * self.block_size, y_blocks * self.block_size))
        self.snake_body = [] # List of tuples
        self.observation = None
        self.current_direction = 0
        self.generate_snake_starting_positions_directions()
        self.fps = pygame.time.Clock()
        self.is_running = True
        self.food_position = self.generate_food()
        self.game_loop(self.window)



    def game_loop(self, window):
        while self.is_running:

            # Draws the surface object to the screen.
            pygame.display.update()
            self.draw_game_state(window=window)


            for event in pygame.event.get():
                # Manually selecting actions
                if event.type == pygame.KEYUP:
                    # Handle the specific key(s) you want to detect when they are released
                    if event.key == pygame.K_0:
                        action = 0
                    elif event.key == pygame.K_1:
                        action = 1
                        pass
                    elif event.key == pygame.K_2:
                        action = 2
                        pass
                    self.env_step(action)


                if event.type == pygame.QUIT:
                    self.is_running = False
                    break
                    pass

            #self.fps.tick(5)  # Adjust the speed of the game here (higher value -> faster)
        pygame.quit()
        sys.exit()

    def env_step(self, action):
        if action == 0:
            # Turn left
            self.current_direction = self.current_direction - 1
            if self.current_direction == -1:
                self.current_direction = 3
        elif action == 1:
            # Turn right
            self.current_direction = self.current_direction + 1
            if self.current_direction == 4:
                self.current_direction = 0
        elif action == 2: #Do nothing
            pass

        self.move_snake()
        pass

    def move_snake(self):
        # Find the new snake head position

        # Find the new snake head position
        head_x, head_y = self.snake_body[0]

        if self.current_direction == 0:  # North
            # Check if next cell is food2
            head_y -= 1
        elif self.current_direction == 1:  # East
            head_x += 1
        elif self.current_direction == 2:  # South
            head_y += 1
        elif self.current_direction == 3:  # West
            head_x -= 1


        # Create a new head position and add it to the front of the list
        new_head = (head_x, head_y)
        if new_head == self.food_position:
            print("Eating food")
            self.snake_body.insert(0, new_head)
            self.food_position = self.generate_food()
        else:
            self.snake_body.insert(0, new_head)
            # Remove the last element to maintain the snake length
            self.snake_body.pop()
            pass


    def generate_snake_starting_positions_directions(self):
        head_x = random.randint(0, self.x_blocks-1)
        head_y = random.randint(0, self.y_blocks-2)
        self.snake_body.append((head_x, head_y))
        self.snake_body.append((head_x, head_y+1))
        self.current_direction = random.randint(0, 3)

    def generate_food(self):
        all_possible_coordinates = set(
            (x, y) for x in range(self.x_blocks) for y in range(self.y_blocks)
        )

        # Remove coordinates occupied by the snake's body from the set
        empty_coordinates = all_possible_coordinates - set(self.snake_body)

        # Randomly select one of the remaining empty coordinates for the food
        food_x, food_y = random.choice(list(empty_coordinates))

        return food_x, food_y

    def draw_game_state(self, window):
        # Fill the screen with white color
        window.fill((0, 0, 0))
        # Draw the snake
        for index, (x, y) in enumerate(self.snake_body):
            if index == 0:
                pygame.draw.rect(window, (255, 0, 0), [x * self.block_size, y * self.block_size, self.block_size, self.block_size], 0)
            else:
                pygame.draw.rect(window, (0, 0, 255), [x * self.block_size, y * self.block_size, self.block_size, self.block_size], 0)

        # Draw the food
        pygame.draw.rect(window, (0, 255, 0), [self.food_position[0] * self.block_size, self.food_position[1] * self.block_size, self.block_size, self.block_size], 0)

    def sample_action(self):
        random_action = random.randint(0, 1)
        return random_action