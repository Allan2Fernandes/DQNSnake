import random
import sys
import pygame
import numpy as np
from AgentNetwork.DQN_conv import DQN_conv as DQN
import torch
from collections import deque

class SnakeEnvironment:
    def __init__(self, x_blocks, y_blocks, block_size, device, num_episodes, max_time_steps, batch_size, num_features, render_mode, with_hidden_layer, num_filters, model_directory):
        self.food_eaten = None
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps
        self.device = device
        self.batch_size = batch_size
        self.render_mode = render_mode
        self.food_position = None
        self.is_running = None
        self.current_direction = None
        self.snake_body = None
        self.block_size = block_size
        self.x_blocks = x_blocks
        self.y_blocks = y_blocks
        self.action_size = 3
        self.state_size = self.x_blocks*self.y_blocks + 8
        self.dqn = DQN(state_size=self.state_size,
                       action_size=self.action_size,
                       device=device,
                       num_features=num_features,
                       num_channels=13,
                       with_hidden_layer=with_hidden_layer,
                       num_filters=num_filters,
                       model_directory=model_directory
                       )
        pygame.init()
        # create the display surface object of specific dimension.
        if render_mode == 'Human':
            self.window = pygame.display.set_mode((x_blocks * self.block_size, y_blocks * self.block_size))

        self.fps = pygame.time.Clock()
        self.game_loop()

    def expand_state_dims(self, state):
        state = torch.tensor(state)
        state = torch.unsqueeze(state, dim=0)
        state = state.to(self.device)
        return state

    def preprocess_state_for_conv_net(self, state):
        state = torch.unsqueeze(state, dim=0)
        state = torch.permute(state, (0, 3, 1, 2))
        state = state.to(self.device)
        state = torch.tensor(state, dtype=torch.float32)
        return state


    def reset(self):
        self.snake_body = []  # List of tuples
        self.current_direction = 0
        self.generate_snake_starting_positions_directions()
        self.is_running = True
        self.food_position = self.generate_food()
        return self.get_state_conv(False)

    def game_loop(self):
        num_steps_completed = 0
        running_scores = deque(maxlen=100)
        for episode in range(self.num_episodes):
            init_state = self.reset()
            state = self.preprocess_state_for_conv_net(init_state)
            done = False
            episode_reward = 0
            time_steps_without_reward = 0
            while not done:
                num_steps_completed += 1

                if num_steps_completed % self.dqn.update_rate == 0:
                    self.dqn.update_target_network()
                    pass

                # Draws the surface object to the screen.
                if self.render_mode == "Human":
                    pygame.display.update()
                    self.draw_game_state(window=self.window)
                    self.window.blit(self.text_surface, (0,0))
                    # fps tick here
                    #self.fps.tick(2)
                    pass

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
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

                next_state = self.preprocess_state_for_conv_net(next_state)
                self.dqn.store_transition(state, action, reward, next_state, done)
                # Store transition
                state = next_state
                if done:
                    print("Episode {}, {:.2f}, Food eaten: {}, 100 game running average = {}".format(episode, episode_reward, self.food_eaten, np.average(running_scores)))
                    if self.food_eaten > 27:
                        self.dqn.save_entire_model(episode=episode)
                    pass

                if len(self.dqn.replay_buffer) > self.batch_size:
                    self.dqn.train_double_DQN(batch_size=self.batch_size)
                    pass
                if time_steps_without_reward >= self.max_time_steps:
                    print("truncated")
                    done = True
                pass
            self.dqn.decay_epsilon()
            running_scores.append(episode_reward)
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
        elif action == 2: # Continue straight
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
            self.food_eaten += 1
            reward += 1
        else:
            #reward -= 0.001
            # Remove the last element to maintain the snake length
            self.snake_body.pop()
            # Check if the snake eats itself or meets edge
            if (head_x, head_y) in self.snake_body:
                reward -= 1
                done = True
            elif head_x == -1 or head_x == self.x_blocks or head_y == -1 or head_y == self.y_blocks:
                reward -= 1
                done = True
                pass

            self.snake_body.insert(0, new_head)
            pass
        next_state = self.get_state_conv(done)
        return next_state, reward, done


    def generate_snake_starting_positions_directions(self):
        self.food_eaten = 0
        head_x = random.randint(1, self.x_blocks-1)
        head_y = random.randint(1, self.y_blocks-1)
        self.snake_body.append((head_x, head_y))
        # Attach a possible head to the snake
        # Generate a random number of tail elements
        for _ in range(random.randint(0, 0)): self.attach_new_head_to_snake()
        self.current_direction = random.randint(0, 3)
        pass

    def attach_new_head_to_snake(self):
        head_x, head_y = self.snake_body[0]
        # Possible positions are (x,y-1),(x+1,y), (x, y+1), (x-1, y)
        possible_positions = [(head_x, head_y - 1), (head_x + 1, head_y), (head_x, head_y + 1), (head_x - 1, head_y)]

        while True:
            if not possible_positions:

                # If the list is empty, no possible position found, break the loop
                break

            new_snake_position = possible_positions[random.randint(0, len(possible_positions) - 1)]
            new_head_x, new_head_y = new_snake_position

            if new_head_x <= -1 or new_head_x >= self.x_blocks or new_head_y <= -1 or new_head_y >= self.y_blocks or new_snake_position in self.snake_body:
                # Remove the conflicting position from possible_positions
                possible_positions.remove(new_snake_position)
            else:
                # A valid position found, break the loop
                self.snake_body.insert(0, new_snake_position)
                break


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
            game_state = np.pad(game_state, (0, 8), 'constant')
            return game_state
        for x,y in self.snake_body:
            game_state[y][x] = -1 # all the tail elements
            pass

        game_state[self.snake_body[0][1]][self.snake_body[0][0]] = 1
        #game_state[self.food_position[1]][self.food_position[0]] = 3
        game_state = game_state.flatten()
        game_state = np.pad(game_state, (0, 8), 'constant')
        # Relative food position
        game_state[-7] = (self.snake_body[0][0] - self.food_position[0]) < 0 # Head is to the left of food
        game_state[-6] = (self.snake_body[0][0] - self.food_position[0]) > 0 # Head is to the right of food
        game_state[-5] = (self.snake_body[0][1] - self.food_position[1]) < 0 # Head is below food
        game_state[-4] = (self.snake_body[0][1] - self.food_position[1]) > 0 # Head is above food
        # Direction
        direction_one_hot = torch.nn.functional.one_hot(torch.tensor(self.current_direction), 4)
        game_state[-4] = direction_one_hot[0].item() # North
        game_state[-3] = direction_one_hot[1].item() # East
        game_state[-2] = direction_one_hot[2].item() # South
        game_state[-1] = direction_one_hot[3].item() # West

        return game_state

    def get_state_conv(self, done):
        game_state = np.zeros((self.x_blocks, self.y_blocks))
        if done:
            game_state = np.expand_dims(game_state, axis=-1)
            padded_filters = np.zeros((self.x_blocks, self.y_blocks, 4))
            game_state = torch.tensor(game_state)
            padded_filters = torch.tensor(padded_filters)
            game_state = torch.cat((game_state, padded_filters), dim=-1)
            return game_state

        for x,y in self.snake_body:
            game_state[x][y] = -1
            pass
        game_state[self.snake_body[0][0]][self.snake_body[0][1]] = 1

        # Apply the direction filters ------------
        direction_one_hot = torch.nn.functional.one_hot(torch.tensor(self.current_direction), 4)
        # Reshape the initial tensor to have dimensions (1, 1, 4)
        reshaped_tensor = direction_one_hot.unsqueeze(0).unsqueeze(1)
        reshaped_tensor = reshaped_tensor.repeat(self.x_blocks, self.y_blocks, 1)
        game_state = np.expand_dims(game_state, axis=-1)
        game_state = torch.tensor(game_state)
        game_state = torch.cat((game_state, reshaped_tensor), dim=-1)

        # Apply distance to food filters ---------
        x1 = 1 if ((self.snake_body[0][0] - self.food_position[0]) < 0) else 0
        x2 = 1 if ((self.snake_body[0][0] - self.food_position[0]) > 0) else 0
        y1 = 1 if ((self.snake_body[0][1] - self.food_position[1]) < 0) else 0
        y2 = 1 if ((self.snake_body[0][1] - self.food_position[1]) > 0) else 0
        relative_distance_vector = torch.tensor([x1, x2, y1, y2])
        reshaped_tensor2 = relative_distance_vector.unsqueeze(0).unsqueeze(1)
        reshaped_tensor2 = reshaped_tensor2.repeat(self.x_blocks, self.y_blocks, 1)
        game_state = torch.cat((game_state, reshaped_tensor2), dim=-1)

        # Apply the danger filters ---------------
        head_x, head_y = self.snake_body[0]
        # Danger above
        danger_above = 1 if head_y-1 < 0 or (head_x, head_y - 1) in self.snake_body else 0
        # Danger below
        danger_below = 1 if head_y+1 >= self.y_blocks or (head_x, head_y + 1) in self.snake_body else 0
        # Danger East
        danger_right = 1 if head_x+1 >= self.x_blocks or (head_x + 1, head_y) in self.snake_body else 0
        # Danger west
        danger_left = 1 if head_x-1 < 0 or (head_x-1, head_y) in self.snake_body else 0
        danger_directions_tensor = torch.tensor([danger_above, danger_below, danger_left, danger_right])
        reshaped_tensor3 = danger_directions_tensor.unsqueeze(0).unsqueeze(1)
        reshaped_tensor3 = reshaped_tensor3.repeat(self.x_blocks, self.y_blocks, 1)
        game_state = torch.cat((game_state, reshaped_tensor3), dim=-1)
        return game_state



    def draw_game_state(self, window):
        # Fill the screen with white color
        window.fill((0, 0, 0))
        # Draw the snake
        for index, (x, y) in enumerate(self.snake_body):
            if index == 0:
                pygame.draw.rect(window, (255, 0, 0), [x * self.block_size, y * self.block_size, self.block_size*0.8, self.block_size*0.8], 0)
            else:
                pygame.draw.rect(window, (0, 0, 255), [x * self.block_size, y * self.block_size, self.block_size*0.8, self.block_size*0.8], 0)

        # Draw the food
        pygame.draw.rect(window, (0, 255, 0), [self.food_position[0] * self.block_size, self.food_position[1] * self.block_size, self.block_size*0.8, self.block_size*0.8], 0)

        # Display the score
        font = pygame.font.Font(None, 36)
        text = "Score: " + str(self.food_eaten)
        text_color = (255, 0, 0)
        self.text_surface = font.render(text, True, text_color)


        pass

    def sample_action(self):
        random_action = random.randint(0, 1)
        return random_action