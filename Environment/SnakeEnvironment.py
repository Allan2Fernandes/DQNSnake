import random
import sys
import pygame
import numpy as np
from Environment.DQN import DQN
import torch

class SnakeEnvironment:
    def __init__(self, x_blocks, y_blocks, block_size, device, num_episodes, max_time_steps, batch_size):
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps
        self.device = device
        self.batch_size = batch_size
        self.food_position = None
        self.is_running = None
        self.current_direction = None
        self.snake_body = None
        self.block_size = block_size
        self.x_blocks = x_blocks
        self.y_blocks = y_blocks
        self.action_size = 3
        self.state_size = self.x_blocks*self.y_blocks + 2
        self.dqn = DQN(action_size=self.action_size, state_size=self.state_size, device=device)
        pygame.init()
        # create the display surface object of specific dimension.
        self.window = pygame.display.set_mode((x_blocks * self.block_size, y_blocks * self.block_size))

        self.fps = pygame.time.Clock()
        self.game_loop(self.window)
        pass

    def expand_state_dims(self, state):
        state = torch.tensor(state)
        state = torch.unsqueeze(state, dim=0)
        state = state.to(self.device)
        return state


    def reset(self):
        self.snake_body = []  # List of tuples
        self.current_direction = 0
        self.generate_snake_starting_positions_directions()
        self.is_running = True
        self.food_position = self.generate_food()
        return self.get_state(False)

    def game_loop(self, window):
        num_steps_completed = 0
        for episode in range(self.num_episodes):
            init_state = self.reset()
            state = self.expand_state_dims(init_state)
            done = False
            time_step = 0
            episode_reward = 0
            time_steps_without_reward = 0
            while not done:
                num_steps_completed += 1
                if num_steps_completed % self.dqn.update_rate == 0:
                    self.dqn.update_target_network()
                # Draws the surface object to the screen.
                pygame.display.update()
                self.draw_game_state(window=window)
                for event in pygame.event.get():
                    # if event.type == pygame.KEYUP:
                    #     if event.key == pygame.K_0:
                    #         action = 0
                    #     elif event.key == pygame.K_1:
                    #         action = 1
                    #     elif event.key == pygame.K_2:
                    #         action = 2
                    #     else:
                    #         action = 2
                    #     next_state, reward, done = self.env_step(action)

                    #next_state = self.expand_state_dims(next_state)
                    #self.dqn.store_transition(state, action, reward, next_state, done, 0)
                    # Store transition
                    #state = next_state
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                        break
                        pass
                    pass

                action = self.dqn.epsilon_greedy(state=state)
                next_state, reward, done = self.env_step(action)
                episode_reward += reward
                if reward == 0:
                    time_steps_without_reward += 1
                else:
                    time_steps_without_reward = 0
                    pass

                next_state = self.expand_state_dims(next_state)
                self.dqn.store_transition(state, action, reward, next_state, done)
                # Store transition
                state = next_state
                self.fps.tick(60)
                if done:
                    print("Episode", episode, episode_reward, "Snake Length", len(self.snake_body))
                    pass

                if len(self.dqn.replay_buffer) > self.batch_size:
                    self.dqn.train_double_DQN(batch_size=self.batch_size)
                    pass
                if time_steps_without_reward >= self.max_time_steps:
                    print("truncated")
                    done = True
                pass
            self.dqn.decay_epsilon()

            pass
        pass


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

        next_state, reward, done = self.move_snake()
        return next_state, reward, done

    def move_snake(self):
        reward = 0
        done = False
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
            self.snake_body.insert(0, new_head)
            # Check if the game has ended and if not, generate food
            if len(self.snake_body) == self.x_blocks*self.y_blocks:
                done = True
                reward += 20
            else:
                self.food_position = self.generate_food()
                pass
            reward += 5
        else:
            reward -= 0.01
            # Remove the last element to maintain the snake length
            self.snake_body.pop()
            # Check if the snake eats itself or meets edge
            if (head_x, head_y) in self.snake_body:
                #print("eating tail")
                reward -= 1
                done = True
            elif head_x == -1 or head_x == self.x_blocks or head_y == -1 or head_y == self.y_blocks:
                #print("edge detected")
                reward -= 1
                done = True
                pass

            self.snake_body.insert(0, new_head)
            pass
        next_state = self.get_state(done)
        return next_state, reward, done


    def generate_snake_starting_positions_directions(self):
        head_x = random.randint(1, self.x_blocks-1)
        head_y = random.randint(1, self.y_blocks-2)
        self.snake_body.append((head_x, head_y))
        self.snake_body.append((head_x, head_y+1))
        self.current_direction = 0

    def generate_food(self):
        all_possible_coordinates = set(
            (x, y) for x in range(self.x_blocks) for y in range(self.y_blocks)
        )

        # Remove coordinates occupied by the snake's body from the set
        empty_coordinates = all_possible_coordinates - set(self.snake_body)

        # Randomly select one of the remaining empty coordinates for the food
        food_x, food_y = random.choice(list(empty_coordinates))

        return food_x, food_y

    def get_state(self, done):
        # Create a 0s matrix the size of the board
        game_state = np.zeros((self.x_blocks, self.y_blocks))
        game_state = game_state.astype(np.float32)
        if done:
            game_state = game_state.flatten()
            game_state = np.pad(game_state, (0, 2), 'constant')
            return game_state
        for x,y in self.snake_body:
            game_state[y][x] = -1
            pass

        game_state[self.snake_body[0][1]][self.snake_body[0][0]] = 1
        game_state[self.food_position[1]][self.food_position[0]] = 3
        game_state = game_state.flatten()
        game_state = np.pad(game_state, (0, 2), 'constant')
        game_state[-2] = (self.snake_body[0][0] - self.food_position[0])/self.x_blocks
        game_state[-1] = (self.snake_body[0][1] - self.food_position[1])/self.y_blocks

        return game_state


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