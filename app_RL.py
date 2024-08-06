import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # This will output whether CUDA/GPU is being used!


# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 30))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect(midbottom=(SCREEN_WIDTH//2, SCREEN_HEIGHT-10))

    def move(self, action):
        if action == 0:  # Left
            self.rect.x = max(0, self.rect.x - 10)
        elif action == 2:  # Right
            self.rect.x = min(SCREEN_WIDTH - self.rect.width, self.rect.x + 10)

class Asteroid(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((20, 20))
        self.image.fill(RED)
        self.rect = self.image.get_rect(midtop=(random.randint(0, SCREEN_WIDTH), 0))
        self.speed = random.randint(5, 10)

    def update(self):
        self.rect.y += self.speed

class SpaceGame:
    def __init__(self):
        self.player = Player()
        self.asteroids = pygame.sprite.Group()
        self.score = 0
        self.frame = 0
        self.game_over = False

    def reset(self):
        self.__init__()
        return self.get_state()

    def step(self, action):
        reward = 0
        self.frame += 1
        self.player.move(action)

        # Spawn new asteroids
        if self.frame % 30 == 0:
            self.asteroids.add(Asteroid())

        # Update and check collision
        self.asteroids.update()
        if pygame.sprite.spritecollide(self.player, self.asteroids, False):
            reward -= 100
            self.game_over = True

        # Remove off-screen asteroids and increase score
        for asteroid in self.asteroids.copy():
            if asteroid.rect.top >= SCREEN_HEIGHT:
                reward += 10
                self.score += 1
                self.asteroids.remove(asteroid)

        reward += 1  # Living reward

        return self.get_state(), reward, self.game_over

    def get_state(self):
        state = [self.player.rect.centerx / SCREEN_WIDTH] + [
            asteroid.rect.centerx / SCREEN_WIDTH if asteroid.rect.centery < SCREEN_HEIGHT//2 else 0
            for asteroid in self.asteroids
        ][:3]
        return np.array(state + [0]*(4-len(state)), dtype=np.float32)

    def render(self):
        screen.fill(BLACK)
        screen.blit(self.player.image, self.player.rect)
        self.asteroids.draw(screen)
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(text, (10, 10))
        pygame.display.flip()

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 3)
        ).to(device)  # This moves the entire sequential to GPU!

    def forward(self, x):
        x = x.to(device)  # Ensure input is on GPU
        return torch.softmax(self.fc(x), dim=-1)


class PPO:
    def __init__(self):
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.002)

    def update(self, states, actions, rewards):
        states = torch.FloatTensor(np.array(states)).to(device)   # Move to GPU
        actions = torch.LongTensor(actions).to(device)  # Move to GPU
        returns = self.compute_returns(rewards).to(device)  # Move to GPU


        for _ in range(5):  # PPO update epochs
            action_probs = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            loss = -(new_log_probs * returns).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_returns(self, rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Move to GPU!
        with torch.no_grad():  # Good practice: no need for grad-tracking during inference
            probs = self.policy(state)
        m = Categorical(probs.cpu())  # Move back to CPU for sampling!
        action = m.sample()
        return action.item()

def train():
    game = SpaceGame()
    agent = PPO()
    
    for episode in range(1000):  # Number of episodes
        state = game.reset()
        states, actions, rewards = [], [], []

        while not game.game_over:
            action = agent.select_action(state)
            next_state, reward, done = game.step(action)
            game.render()
            clock.tick(30)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        agent.update(states, actions, rewards)
        print(f"Episode {episode}, Score: {game.score}")

train()
pygame.quit()

