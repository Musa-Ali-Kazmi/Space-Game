import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import time


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

pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Space Game")
pygame.image.load('plane.png')
pygame.image.load('asteroid.png')


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('plane.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (50, 30))  # Adjust size as needed
        self.rect = self.image.get_rect(midbottom=(SCREEN_WIDTH//2, SCREEN_HEIGHT-10))


    def move(self, action):
        if action == 0:  # Left
            self.rect.x = max(0, self.rect.x - 10)
        elif action == 2:  # Right
            self.rect.x = min(SCREEN_WIDTH - self.rect.width, self.rect.x + 10)

class Asteroid(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('asteroid.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (20, 20))  # Adjust size as needed
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
        self.episode = 0

    def reset(self):
        self.__init__()
        self.episode += 1  # Increment episode number
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
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        episode_text = font.render(f"Episode: {self.episode}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(episode_text, (10, 50))  # Display episode number
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
    
    def save(self, checkpoint):
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load(self, checkpoint):
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
    


class CheckpointManager:
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save_checkpoint(self, agent, optimizer, game, episode, score):
        timestamp = int(time.time())
        filename = f"checkpoint_ep{episode}_score{score}_{timestamp}.pth"
        filepath = os.path.join(self.save_dir, filename)
        
        torch.save({
            'episode': episode,
            'model_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'game_state': game.get_state().tolist()
        }, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_latest_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(self.save_dir, x)))
        filepath = os.path.join(self.save_dir, latest_checkpoint)
        print(f"Loading checkpoint: {filepath}")
        return torch.load(filepath)


def train():
    game = SpaceGame()
    agent = PPO()
    checkpoint_manager = CheckpointManager()
    running = True
    start_episode = 0
    best_score = 0

    # Try to load the latest checkpoint
    checkpoint = checkpoint_manager.load_latest_checkpoint()
    if checkpoint:
        agent.load(checkpoint)
        start_episode = checkpoint['episode']
        game.score = checkpoint['score']
        game.episode = start_episode
        best_score = game.score
        print(f"Resuming from episode {start_episode} with score {game.score}")
    
    for episode in range(start_episode, 100):  # Number of episodes
        if not running:
            break
        
        game.episode = episode + 1
        state = game.reset()
        states, actions, rewards = [], [], []

        while not game.game_over:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    checkpoint_manager.save_checkpoint(agent, agent.optimizer, game, episode, game.score)
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Press 'Q' to quit and save
                        running = False
                        checkpoint_manager.save_checkpoint(agent, agent.optimizer, game, episode, game.score)
                        return
                    elif event.key == pygame.K_s:  # Press 'S' to save manually
                        checkpoint_manager.save_checkpoint(agent, agent.optimizer, game, episode, game.score)

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

        # Save checkpoint if it's the best score
        if game.score > best_score:
            best_score = game.score
            checkpoint_manager.save_checkpoint(agent, agent.optimizer, game, episode, game.score)

        # Auto-save every 10 episodes
        if episode % 10 == 0:
            checkpoint_manager.save_checkpoint(agent, agent.optimizer, game, episode, game.score)



train()
pygame.quit()  


