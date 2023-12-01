import pygame
import sys
import torch
import numpy as np
from constants import colors
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

pygame.init()

WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Slime mold")
species = 2

clock = pygame.time.Clock()

ant_number = 50000
max_speed = 9
rand_speed_weight = 0.1
phero_follow = 1
decay_factor = 0.99
#fader = 200

ticks = 100

perception_angle = 36  # degrees
perception_radius = 40


def sample_directions(ant_positions, pheromone_map, max_speed, perception_radius, perception_angle_deg, ant_speeds):
    num_samples = 3 # Number of sampled directions

    # Convert perception angle to radians
    perception_angle = math.radians(perception_angle_deg)
    
    speed = torch.sqrt(ant_speeds.T[0]**2 + ant_speeds.T[1]**2)

    # Repeat the angles for each ant
    angles = torch.linspace(-perception_angle / 2,
                            perception_angle / 2, num_samples).to(device)
    angles = angles.repeat(ant_positions.shape[0], 1)  # Repeat for each ant

    # Offset angles by ant's direction
    # Offset by ant's direction
    angles += torch.atan2(ant_speeds[:, 1], ant_speeds[:, 0]).unsqueeze(1)
    #angles = (angles + 2 * math.pi) % (2 * math.pi)  # Normalize angles

    # Get positions for sampled directions
    sample_positions = ant_positions[:, None] + torch.max(speed) * \
        torch.stack([torch.cos(angles), torch.sin(angles)], dim=2)

    # Normalize positions to fit the grid range [-1, 1]
    sample_positions = (sample_positions.float() /
                        pheromone_map.shape[0]) * 2 - 1
    
    
    # Calculate pheromone concentrations for each sample
    pheromone_concentrations = torch.nn.functional.grid_sample(
        # Add batch and channel dimensions
        pheromone_map.unsqueeze(0).unsqueeze(0),
        sample_positions.unsqueeze(0),
        align_corners=True
    ).squeeze()  # Remove added dimensions

    # Reshape concentrations back to [num_ants, num_samples]
    pheromone_concentrations = pheromone_concentrations.view(
        ant_positions.shape[0], num_samples)

    # Find the index of the direction with the maximum concentration for each ant
    best_direction_indices = torch.argmax(pheromone_concentrations, dim=1)

    # Retrieve the corresponding direction for each ant
    best_directions = angles[torch.arange(
        angles.size(0)), best_direction_indices]

    # Calculate new speed based on the best direction for each ant
    new_speed = torch.stack([speed * torch.cos(best_directions),
                            speed * torch.sin(best_directions)], dim=1)

    return new_speed


def create_ants(ant_number, max_speed, width, height):
    x_pos = torch.randint(0, width, (ant_number,)).to(device)
    y_pos = torch.randint(0, height, (ant_number,)).to(device)
    pos = torch.stack((x_pos, y_pos), dim=1).float()

    speeds = (torch.rand(ant_number,2).to(device) * max_speed * 2) - max_speed

    return pos, speeds


def update_speed(pos, speed, max_speed, rand_speed_weight, width, height):

    new_speed_x = torch.where((pos[:, 0] <= 0) | (
        pos[:, 0] >= width), -speed[:, 0], speed[:, 0])
    new_speed_y = torch.where((pos[:, 1] <= 0) | (
        pos[:, 1] >= height), -speed[:, 1], speed[:, 1])

    # new_speeds = torch.randn_like(speed).to(device) * max_speed

    # new_speed_x = torch.where((pos[:,0] <= 0) | (pos[:,0] >= width),  new_speeds[:,0], speed[:,0])
    # new_speed_x = torch.where((pos[:,1] <= 0) | (pos[:,1] >= height), new_speeds[:,0], new_speed_x)
    #
    # new_speed_y = torch.where((pos[:,1] <= 0) | (pos[:,1] >= height), new_speeds[:,1], speed[:,1])
    # new_speed_y = torch.where((pos[:,0] <= 0) | (pos[:,0] >= width),  new_speeds[:,1], new_speed_y)

    new_pos_x = torch.clamp(pos[:, 0], 0, width)
    new_pos_y = torch.clamp(pos[:, 1], 0, height)

    new_speed = torch.stack((new_speed_x, new_speed_y), dim=1)
    new_pos = torch.stack((new_pos_x, new_pos_y), dim=1)

    rand_speed = (torch.rand_like(speed).to(device) * max_speed * 2) - max_speed
    new_speed = new_speed * (1 - rand_speed_weight) + \
        rand_speed * rand_speed_weight

    return new_speed, new_pos


def deposit_pheromones(ant_positions, pheromone_map, pheromone_value =0.1):
    rounded_positions = ant_positions.long().clamp(
        min=0)  # Ensure positions are within bounds
    indices = rounded_positions[:, 0] * HEIGHT + rounded_positions[:, 1]
    # Clamp indices to stay within the range
    indices = indices.clamp(max=WIDTH * HEIGHT - 1)
    pheromone_map = pheromone_map.view(-1)
    pheromone_map.scatter_add_(0, indices, torch.ones_like(
        indices, dtype=torch.float32) * pheromone_value)
    pheromone_map = pheromone_map.view(WIDTH, HEIGHT)
    return pheromone_map




def diffuse_pheromones(pheromone_map, kernel = 'k2'):
    # Define a Gaussian kernel for diffusion
    
    if kernel == 'k2':
        kernel = torch.tensor([

                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 0, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],

                                ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        kernel /= kernel.sum()  # Normalize the kernel

        # Apply 2D convolution to diffuse pheromones
        pheromone_map = pheromone_map.unsqueeze(0).unsqueeze(
            0)  # Add batch and channel dimensions
        diffused_map = torch.conv2d(pheromone_map, kernel, padding=2)
    
    if kernel == 'k1':
        kernel = torch.tensor([
                                [1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]
                                ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        kernel /= kernel.sum()  # Normalize the kernel

        # Apply 2D convolution to diffuse pheromones
        pheromone_map = pheromone_map.unsqueeze(0).unsqueeze(
            0)  # Add batch and channel dimensions
        diffused_map = torch.conv2d(pheromone_map, kernel, padding=1)
    
    
    diffused_map = diffused_map.squeeze(0).squeeze(
        0)  # Remove batch and channel dimensions

    return diffused_map


# Assuming pheromone_map has values between 0 and 255
def render_pheromones(screen, pheromone_map, total_pheromones):
    # Scale the pheromone_map to the range [0, 255]
    #scaled_pheromones = torch.clamp(pheromone_map, 0, 255)
    scaled_pheromones = (pheromone_map / torch.max(pheromone_map)) *255
    
    # Clamp values to ensure they are within the valid range [0, 255]
    scaled_pheromones = torch.clamp(scaled_pheromones, 0, 255)
    
    total_pheromones.append(torch.sum(scaled_pheromones).item())
    scaled_pheromones = scaled_pheromones/torch.sum(scaled_pheromones) * np.mean(total_pheromones)
    scaled_pheromones = torch.clamp(scaled_pheromones, 0, 255)
    scaled_pheromones = diffuse_pheromones(scaled_pheromones, 'k1')
    #scaled_pheromones = (pheromone_map / torch.max(pheromone_map)) *255

    # Convert the tensor to a numpy array and transpose for Pygame display
    pheromone_array = scaled_pheromones.cpu().numpy().astype(np.uint8)
    pheromone_array = np.transpose(pheromone_array)

    # Apply colormap
    pheromone_array_color = np.zeros((pheromone_array.shape[0], pheromone_array.shape[1], 3), dtype=np.uint8)

    # Purple color
    pheromone_array_color[:, :, 0] = pheromone_array  # Red channel
    pheromone_array_color[:, :, 1] = 0  # Green channel
    pheromone_array_color[:, :, 2] = pheromone_array  # Blue channel

    # Create a Pygame surface from the numpy array
    pheromone_surface = pygame.surfarray.make_surface(pheromone_array_color)

    # Display the pheromone surface on the screen
    screen.blit(pheromone_surface, (0, 0))
    
    return total_pheromones


positions = []
speeds = []
pheromone_maps = []
total_pheromones = []

for spec in range(species):
    pos, speed = create_ants(ant_number, max_speed, WIDTH, HEIGHT)
    pheromone_map = torch.zeros((WIDTH, HEIGHT)).to(device)
    
    positions.append(pos)
    speeds.append(speed)
    pheromone_maps.append(pheromone_map)
    total_pheromones.append([])

pos, speed = create_ants(ant_number, max_speed, WIDTH, HEIGHT)
pheromone_map = torch.zeros((WIDTH, HEIGHT)).to(device)
total_pheromones = []

running = True
paused = False
add_pheromones = False 
while running:
    clock.tick(ticks)
    
    if not paused:
        speed2 = sample_directions(
            pos, pheromone_map, max_speed, perception_radius, perception_angle, speed)

        speed = speed * (1-phero_follow) + speed2 * phero_follow
        speed, pos = update_speed(pos, speed, max_speed,
                                  rand_speed_weight, WIDTH, HEIGHT)
        pos += speed

        # for i in range(ant_number):
        #    pygame.draw.circle(screen, colors.BROWN, pos[i].tolist(), 3)

        pheromone_map = deposit_pheromones(pos, pheromone_map)
        pheromone_map = diffuse_pheromones(pheromone_map)
        #pheromone_map = diffuse_pheromones(pheromone_map)
        pheromone_map *= decay_factor
        total_pheromones = render_pheromones(screen, pheromone_map, total_pheromones)
    
    if add_pheromones:
        x,y = pygame.mouse.get_pos()
        x = max(0, min(WIDTH -1, x))
        y = max(0, min(HEIGHT -1, y))
        
        pheromone_value = torch.max(pheromone_map)
        pheromone_map = deposit_pheromones(torch.tensor([[y,x],]).to(device),pheromone_map, torch.max(pheromone_map) *5)
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not(paused)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                add_pheromones = True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                add_pheromones = False


    pygame.display.update()

pygame.quit()
sys.exit()



 