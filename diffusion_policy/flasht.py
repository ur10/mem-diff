import pygame
import sys
import random
import time

pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PushT Environment")

# Colors
WHITE = (255,255,255)
GRAY = (120,120,120)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
DOT_COLOR = (0,0,0)

FPS = 60
clock = pygame.time.Clock()

# Light positions
light_centers = [(200,70), (400,70), (600,70)]
light_colors = [RED, BLUE, GREEN]
target_regions = [pygame.Rect(x-40, 150, 80, 40) for (x,_) in light_centers]

# Gray box and agent
box = pygame.Rect(350, 220, 100, 60)
dot = pygame.Rect(380, 340, 25, 25)  # The agent

# State
current_light_idx = random.randint(0,2)
flash_time = time.time()
flash_duration = 1.0  # seconds
wait_k = 2.0          # k seconds after flash to allow action
k_reached = False

font = pygame.font.SysFont(None,30)

def flash_light(idx):
    pygame.draw.circle(screen, light_colors[idx], light_centers[idx], 32)
    for j in range(3):
        if j != idx:
            pygame.draw.circle(screen, (60,60,60), light_centers[j], 32)

def draw_targets():
    for rect in target_regions:
        pygame.draw.rect(screen, GREEN, rect, 3)

def reset():
    global current_light_idx, flash_time, k_reached, box, dot
    current_light_idx = random.randint(0,2)
    flash_time = time.time()
    k_reached = False
    box.x, box.y = 350, 220
    dot.x, dot.y = 380, 340

# Main loop
while True:
    screen.fill(WHITE)
    draw_targets()
    flash_light(current_light_idx)
    pygame.draw.rect(screen, GRAY, box)
    pygame.draw.ellipse(screen, DOT_COLOR, dot)
    
    # Notification
    now = time.time()
    if now - flash_time < flash_duration:
        info = font.render("Light Flashing!", True, BLACK)
    elif now - flash_time < flash_duration + wait_k:
        info = font.render("Wait for k seconds...", True, BLACK)
    else:
        info = font.render("Move the dot to push the box into green region!", True, BLACK)
        k_reached = True
    screen.blit(info, (30, HEIGHT-35))

    # Move agent after k seconds
    keys = pygame.key.get_pressed()
    speed = 4
    if k_reached:
        if keys[pygame.K_LEFT]:
            dot.x -= speed
        if keys[pygame.K_RIGHT]:
            dot.x += speed
        if keys[pygame.K_UP]:
            dot.y -= speed
        if keys[pygame.K_DOWN]:
            dot.y += speed

        # Check for pushing the box
        if dot.colliderect(box):
            bx_dir = 1 if dot.centerx > box.centerx else -1
            by_dir = 1 if dot.centery > box.centery else -1
            box.x += bx_dir * speed
            box.y += by_dir * speed

    # Win condition
    if box.colliderect(target_regions[current_light_idx]):
        win_text = font.render("Good job! Resetting...", True, BLACK)
        screen.blit(win_text, (300, 10))
        pygame.display.flip()
        pygame.time.wait(1200)
        reset()
        continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.flip()
    clock.tick(FPS)
