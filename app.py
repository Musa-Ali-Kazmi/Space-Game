import pygame
import random
import math
from pygame import mixer

# initialize pygame
pygame.init()


# create window
screen = pygame.display.set_mode((500, 700))

# Start window
start_background1 = pygame.image.load('planets.png').convert_alpha()
start_background2 = pygame.image.load('saturn.png').convert_alpha()
start_background3 = pygame.image.load('uranus.png').convert_alpha()
start_background4 = pygame.image.load('planet.png').convert_alpha()
start = False
font_start = pygame.font.Font('freesansbold.ttf',30)
start_textX = 78
start_textY = 330


#Game_over window
game_over_background1 = pygame.image.load('nuclear-bomb.png').convert_alpha()
game_over = False
game_over_textX = 160
game_over_textY = 300


#sound
mixer.music.load('background.wav')
mixer.music.play(-1)

#Score
score_value = 0

#lives
lives = 10
font_lives = pygame.font.Font('freesansbold.ttf',20)


# Background
background = pygame.image.load('background.png').convert()
background1 = pygame.image.load('background1.png').convert()
backgroundY = 0
background1Y = -1000

# Change icon and name
pygame.display.set_caption("Fly Away")
moto_icon = pygame.image.load('motorcycle.png').convert_alpha()
pygame.display.set_icon(moto_icon)

# Player
player = pygame.image.load('spaceship.png').convert_alpha()
playerX = 220
playerY = 400
player_changeX = 0
player_changeY = 0

# Enemy
enemy = pygame.image.load('ufo.png').convert_alpha()
enemyX = random.randint(0, 436)
enemyY = -100
enemy_changeX = 0
enemy_changeY = 0.3

# Bullet
bullet = pygame.image.load('bullet.png').convert_alpha()
bulletX = 240
bulletY = 350
bullet_changeX = 0
bullet_changeY = 0.9
bullet_state = False


def Player(x, y):
    screen.blit(player, (x, y))


def Enemy(x, y):
    screen.blit(enemy, (x, y))


def Fire_bullet(x, y):
    global bullet_state
    screen.blit(bullet, (x+16, y-30))
    bullet_state = True


def isCollision(enemyX, enemyY, bulletX, bulletY):
    distance = math.sqrt((math.pow(enemyX-bulletX,2)) + (math.pow(enemyY-bulletY,2)))
    if distance < 34:
        return True
    else:
        return False
def isCollision1(enemyX, enemyY, bulletX, bulletY):
    distance = math.sqrt((math.pow(enemyX-bulletX,2)) + (math.pow(enemyY-bulletY,2)))
    if distance < 54:
        return True
    else:
        return False


def start_game(x,y):
    start_text = font_start.render('Press Space to START!!',True,(255,199,25))
    screen.blit(start_text,(x,y))


def end_game(x,y):
    global score_value
    game_over_text = font_start.render('Game Over!',True,(255,199,25))
    retry_text = font_start.render('Press Escape to Retry',True,(255,199,25))
    score_text = font_start.render('Your Score: '+str(score_value),True,(255,199,25))
    screen.blit(game_over_text, (x, y))
    screen.blit(score_text, (x-8, y+50))
    screen.blit(retry_text, (x-65, y+100))


# While Loop
running = True
while running:
    if start and not game_over:
        screen.fill((0, 255, 255))
        screen.blit(background, (0, backgroundY))
        screen.blit(background1, (0, background1Y))
        lives_text = font_lives.render('Lives='+str(lives),True,(255,199,25))
        score_text1 = font_lives.render('Score='+str(score_value),True,(255,199,25))
        screen.blit(lives_text,(0,7))
        screen.blit(score_text1,(400,7))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Keystrokes
            if event.type == pygame.KEYDOWN:

                # Left / Right
                if event.key == pygame.K_LEFT:
                    player_changeX = -0.9
                if event.key == pygame.K_RIGHT:
                    player_changeX = 0.9

                # Up / Down
                if event.key == pygame.K_UP:
                    player_changeY = -0.9
                if event.key == pygame.K_DOWN:
                    player_changeY = 0.9
                # Space
                if event.key == pygame.K_SPACE:
                    if bullet_state is False:
                        Fire_bullet(playerX, playerY)
                        bulletX = playerX
                        bulletY = playerY
                        bullet_sound = mixer.Sound('laser.wav')
                        bullet_sound.play()

            if event.type == pygame.KEYUP:
                # Left / Right
                if event.key == pygame.K_LEFT:
                    player_changeX = 0
                if event.key == pygame.K_RIGHT:
                    player_changeX = 0
                # Up/ Down
                if event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                    player_changeY = 0

        # player
        if playerX > 436:
            playerX = 436
        if playerX < 0:
            playerX = 0
        playerX += player_changeX
        playerY += player_changeY

        # enemy

        enemyY += enemy_changeY

        if enemyY > 700:
            enemyY = -100
            enemyX = random.randint(0, 436)
            enemy_changeY += 0.03
            lives-=1

        # background
        backgroundY += 0.1
        background1Y += 0.1
        if backgroundY > 700:
            backgroundY = -1000
        if background1Y > 700:
            background1Y = -1000

        Player(playerX, playerY)

        if bulletY <= 0:
            bullet_state = False
        if bullet_state:
            Fire_bullet(bulletX, bulletY)
            bulletY -= bullet_changeY

        # Collision
        collision_e_b = isCollision(enemyX, enemyY, bulletX, bulletY)
        collision_e_p = isCollision1(enemyX, enemyY, playerX, playerY)
        if collision_e_b:
            bulletY = 350
            bullet_state = False
            enemyY = -100
            enemyX = random.randint(0, 436)
            enemy_changeY += 0.03
            score_value +=1
            explosion_sound = mixer.Sound('explosion.wav')
            explosion_sound.play()
        if collision_e_p or lives == 0:
            game_over = True

            playerX = 220
            playerY = 400
            player_changeX = 0
            player_changeY = 0

            enemyX = random.randint(0, 436)
            enemyY = -100
            enemy_changeX = 0
            enemy_changeY = 0.3

            bulletX = 240
            bulletY = 350
            bullet_changeX = 0
            bullet_changeY = 0.8
            bullet_state = False

            game_over_sound = mixer.Sound('you_lose.wav')
            game_over_sound.play()


        Enemy(enemyX, enemyY)
    elif not start:
        screen.fill((25, 50, 60))
        screen.blit(start_background1, (100, 400))
        screen.blit(start_background2, (300, 490))
        screen.blit(start_background3, (80, 170))
        screen.blit(start_background4, (260, 70))
        start_game(start_textX , start_textY)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    start = True
    elif game_over:
        screen.fill((25, 25, 2))
        screen.blit(game_over_background1 , (180,500))
        end_game(game_over_textX , game_over_textY )
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    start = False
                    game_over = False
                    score_value = 0
                    lives = 10


    pygame.display.update()
#