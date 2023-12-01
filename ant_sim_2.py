import pygame
import sys
import torch
from constants import colors
import numpy as np
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


ticks = 60
WIDTH, HEIGHT = 800, 800

species = 4
ants_per_species = 1500

pheromone_weight = 0.99
randomness_weight = 0.01

max_speed = 20

seight_angle_deg = 45  # degrees
seight_directions = 4 #1+ number of direction an entity could go
seight_radius = 20

ant_size = 3
evaporation_factor = 0.01


color = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
]


pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ant pheromones")


pos_ants = torch.rand(size=[species, ants_per_species, 2]).to(
    device) * torch.tensor([WIDTH, HEIGHT]).to(device)
speed_ants = (torch.rand(size=[species, ants_per_species, 2]).to(
    device) * 2 - 1) * max_speed
ids = torch.arange(species * ants_per_species).to(device)
ids_pairs = torch.combinations(ids, 2)#.to(device)
original_shape = (species, ants_per_species, 2)
pheromone_maps = torch.zeros([species, WIDTH, HEIGHT]).to(device)
seight_angle_rad = torch.deg2rad(torch.tensor(seight_angle_deg).to(device))

def get_addition_array_for_pheromone_concentrations(species, individuals_per_species):
    result = -torch.ones(species, species * individuals_per_species).to(device)
    for i in range(species):
        result[i, i * ants_per_species: (i + 1) * individuals_per_species] = 1
    return result


def get_deltad2_pairs(pos, ids_pairs) -> torch.Tensor:
    """Calculates the squared distance between different particles
    torch.cdist calculates the distances including itself

    Args:
        pos (tensor): _description_
        ids_pairs (tensor): _description_

    Returns:
        _type_: _description_
    """

    dx = torch.diff(torch.stack(
        [pos[0][ids_pairs[:, 0]], pos[0][ids_pairs[:, 1]]]).T).squeeze()
    dy = torch.diff(torch.stack(
        [pos[1][ids_pairs[:, 0]], pos[1][ids_pairs[:, 1]]]).T).squeeze()
    return dx**2 + dy**2


def update_velocities_collisions(velocities, positions, ids_pairs_collide):
    """updates velocities based on collisions with each particle

    Args:
        velocities (torch.tensor): x_y_velocities
        positions (torch.tensor)): x_y_positions
        ids_pairs_collide (torch.tensor)): ids of colliding particles

    Returns:
        torch.tensor: x_velocity, y_velocity
    """
    velocity1 = velocities[:, ids_pairs_collide[:, 0]]
    velocity2 = velocities[:, ids_pairs_collide[:, 1]]
    position1 = positions[:, ids_pairs_collide[:, 0]]
    position2 = positions[:, ids_pairs_collide[:, 1]]
    distance = position1 - position2
    vel_diff = velocity1 - velocity2
    velocity1_new = velocity1 - \
        torch.sum(vel_diff*distance, axis=0) / \
        torch.sum((distance)**2, axis=0) * distance
    velocity2_new = velocity2 - \
        torch.sum(vel_diff*distance, axis=0) / \
        torch.sum((distance)**2, axis=0) * -distance
    return velocity1_new, velocity2_new


def update_velocity_wall(velocity, position, width, height, radius):
    """Updates velocities if a wall collision is occuring

    Args:
        velocity (tensor): current velocity
        position (tensor): current position
        width (int): width of the screen
        height (in): height of the screen

    Returns:
        _type_: tensor of updated velocities
    """
    velocity[0, position[0] > width - radius] = - \
        torch.abs(velocity[0, position[0] > width - radius])
    velocity[0, position[0] < 0 +
             radius] = torch.abs(velocity[0, position[0] < 0 + radius])
    velocity[1, position[1] > height - radius] = - \
        torch.abs(velocity[1, position[1] > height - radius])
    velocity[1, position[1] < 0 +
             radius] = torch.abs(velocity[1, position[1] < 0 + radius])
    return velocity


def handle_collisions(positions: torch.tensor, speeds: torch.tensor, ids_pairs: torch.tensor, original_shape, height, width, radius: int = 2):
    position_reshaped = positions.reshape(-1, 2).mT
    speed_reshaped = speeds.reshape(-1, 2).mT

    distances_squared = get_deltad2_pairs(position_reshaped, ids_pairs)
    ids_pairs_collide = ids_pairs[distances_squared < (2*radius)**2]

    speed_reshaped[:, ids_pairs_collide[:, 0]], speed_reshaped[:, ids_pairs_collide[:, 1]
                                                               ] = update_velocities_collisions(speed_reshaped, position_reshaped, ids_pairs_collide)
    speed_reshaped = update_velocity_wall(
        speed_reshaped, position_reshaped, width, height, radius)

    position_reshaped += speed_reshaped

    position = position_reshaped.mT.view(original_shape)
    speed = speed_reshaped.mT.view(original_shape)
    return position, speed


def deposit_pheromones(ant_positions: torch.tensor, pheromone_maps: torch.tensor, pheromone_value=1):
    """deposits pheromones at the ant positions

    Args:
        ant_positions (tensor): x,y postions of ants shape[species, x,y]
        pheromone_maps (tensor): map of shape [species, x, y]
        pheromone_value (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: map of shape [species, x, y]
    """
    rounded_positions = ant_positions.long().clamp(min=0)
    indices = rounded_positions[:, :, 0] * HEIGHT + rounded_positions[:, :, 1]
    indices = indices.clamp(max=WIDTH * HEIGHT - 1)
    indices = indices.view(ant_positions.size(0), -1)

    pheromone_maps = pheromone_maps.view(pheromone_maps.size(0), -1)
    pheromone_maps.scatter_add_(1, indices, torch.ones_like(
        indices, dtype=torch.float32).to(device) * pheromone_value)

    pheromone_maps = pheromone_maps.view(pheromone_maps.size(0), WIDTH, HEIGHT)

    return pheromone_maps


def diffuse_pheromones(pheromone_map: torch.Tensor, kernel: str = 'k2'):
    """Diffuses pheromones for each species using the specified kernel

    Args:
        pheromone_map (torch.Tensor): Map of shape [species, x, y]
        kernel (str, optional): Kernel for diffusion of size k1 or k2. Defaults to 'k2'.

    Returns:
        torch.Tensor: Map of shape [species, x, y] with diffused pheromones for each species
    """

    species_count = pheromone_map.size(0)
    diffused_maps = []

    for species_idx in range(species_count):
        species_pheromone_map = pheromone_map[species_idx]

        if kernel == 'k2':
            kernel_matrix = torch.tensor([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(species_pheromone_map.device)
            kernel_matrix /= kernel_matrix.sum()  # Normalize the kernel

            # Apply 2D convolution to diffuse pheromones for the current species
            diffused_map_species = torch.nn.functional.conv2d(
                species_pheromone_map.unsqueeze(0).unsqueeze(0),
                kernel_matrix, padding=2)[0][0]

        elif kernel == 'k1':
            kernel_matrix = torch.tensor([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(species_pheromone_map.device)
            kernel_matrix /= kernel_matrix.sum()  # Normalize the kernel

            # Apply 2D convolution to diffuse pheromones for the current species
            diffused_map_species = torch.nn.functional.conv2d(
                species_pheromone_map.unsqueeze(0).unsqueeze(0),
                kernel_matrix, padding=1)[0][0]

        diffused_maps.append(diffused_map_species)

    return torch.stack(diffused_maps)


def render_pheromones(screen: pygame.display, pheromone_maps: torch.Tensor, total_pheromones: torch.Tensor, colors):
    """Renders pheromone maps on a pygame display

    Args:
        screen (pygame.display): instance of a pygame display
        pheromone_map (torch.Tensor): Map of shape [species, x, y]
        total_pheromones (torch.Tensor): []
        colors (list): List of color indices for RGB channels

    Returns:
        torch.Tensor: Map of shape [species, x, y]
    """
    max_pheromone_species = torch.max(pheromone_maps.view(
        pheromone_maps.size(0), -1), dim=1)[0].view(-1, 1, 1)
    scaled_pheromones = (pheromone_maps / max_pheromone_species) * 255
    scaled_pheromones = torch.clamp(scaled_pheromones, 0, 255)
    scaled_pheromones = diffuse_pheromones(scaled_pheromones, 'k1')

    pheromone_array = scaled_pheromones.cpu().numpy().astype(np.uint8)

    # Create an empty array to combine all species' maps
    combined_map = np.zeros(
        (pheromone_array.shape[1], pheromone_array.shape[2], 3), dtype=np.uint8)

    # Combine maps using different colors for each species
    for species_idx in range(pheromone_array.shape[0]):
        # Select a color for the current species
        color = colors[species_idx]
        color_channels = [0, 1, 2]  # Assuming RGB channels

        # Apply the color to the current species' map and add it to the combined map
        for channel_idx, color_channel in enumerate(color_channels):
            combined_map[:, :, color_channel] += pheromone_array[species_idx] * \
                color[channel_idx]

    # Clamp the combined map values to ensure they are within the valid range [0, 255]
    combined_map = np.clip(combined_map, 0, 255).astype(np.uint8)

    # Create a Pygame surface from the combined map
    pheromone_surface = pygame.surfarray.make_surface(combined_map)

    # Display the pheromone surface on the screen
    screen.blit(pheromone_surface, (0, 0))

    return total_pheromones


def cartesian_to_polar(x_y_coordinates: torch.tensor) -> torch.tensor:
    """changes cartesian coordanates into ppolar coordinates

    Args:
        x_y_coordinates (torch.tensor): tensor of caretesian coordinates shape[:,:,2]

    Returns:
        torch.tensor: tensor of polar coordinates shape[:,:,2]
    """
    x = x_y_coordinates[:, :, 0]
    y = x_y_coordinates[:, :, 1]
    radius = torch.norm(x_y_coordinates, dim=2).to(device)
    angle = torch.atan2(y, x).to(device)
    return angle, radius


def polar_to_cartesian(angles, magnitudes):
    x = magnitudes * torch.cos(angles)
    y = magnitudes * torch.sin(angles)
    return torch.stack((x, y), axis=-1)


def get_sector_angles(seight_angle: int, seight_directions: int, species: int = 1, individuals_per_species: int = 1) -> torch.Tensor:
    """Generates a tensor of shape [species, individuals_per_species, seight directions]
        seight directions contains possible lines of seight

    Args:
        seight_angle (int): _description_
        seight_directions (int): _description_
        species (int, optional): _description_. Defaults to 1.
        individuals_per_species (int, optional): _description_. Defaults to 1.

    Returns:
        torch.Tensor: shape [species, individuals_per_species, seight directions]
    """
    sector_angle = seight_angle / (seight_directions - 1)
    start_angle = - seight_angle / 2
    angles = torch.arange(start_angle, start_angle +
                          seight_angle, sector_angle).to(device)
    return angles.unsqueeze(0).unsqueeze(0).expand(species, individuals_per_species, -1)


def calc_area_circle_segment(radius, angle):
    return 0.5 * radius**2 * (angle - torch.sin(angle))


def get_sample_angles(sector_angles, speed_angles):
    expanded_speed_angles = speed_angles.unsqueeze(-1).expand_as(sector_angles)
    return expanded_speed_angles + sector_angles


def get_sample_positions(coordinates: torch.Tensor, angles: torch.Tensor, pheromone_map: torch.Tensor, seight_radius: int):
    # Shape: [species, ants_per_species, angles, (x, y)]
    rel_pos = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    # Expand dimensions to match the shape of coordinates
    # Shape: [species, ants_per_species, 1, (x, y)]
    coordinates = coordinates.unsqueeze(-2)

    # Calculate sample positions
    sample_positions = coordinates + seight_radius * rel_pos
    return sample_positions


def get_pheromone_concentrations(sample_positions, pheromone_maps, addition_array_for_pheromone_concentrations):
    species, ants_per_species, positions, _ = sample_positions.shape


    sample_positions = (sample_positions.float() /
                    torch.tensor([pheromone_maps.shape[1], pheromone_maps.shape[2]]).to(device)) * 2 - 1
    sample_positions_reshaped = sample_positions.view(1, -1, positions, 2)
    
    pheromone_concentrations = torch.zeros([species, ants_per_species, positions]).to(device)
    
    
    for i, pheromone_map in enumerate(pheromone_maps):
        
        pheromone_concentrations_species = torch.nn.functional.grid_sample(
            pheromone_map.unsqueeze(0).unsqueeze(0),
            sample_positions_reshaped,
            align_corners=True
        )
        
        expanded_array1 = addition_array_for_pheromone_concentrations[i].view(1, 1, species*ants_per_species, 1).expand_as(pheromone_concentrations_species)
        pheromone_concentrations_species = torch.mul(expanded_array1, pheromone_concentrations_species)
        pheromone_concentrations_species = pheromone_concentrations_species.view(species, ants_per_species, positions)
        pheromone_concentrations += pheromone_concentrations_species

    return pheromone_concentrations

def get_best_direction_indecies(tensor):
    max_indecies = torch.argmax(tensor, dim=2)
    return max_indecies



def calculate_new_speed(speed_ants, pos_ants ):
    sector_angles = get_sector_angles(
        seight_angle_rad, seight_directions, species, ants_per_species)
    speed_angels, speed_magnitute = cartesian_to_polar(speed_ants)
    sample_angles = get_sample_angles(sector_angles, speed_angels)

    sample_positions = get_sample_positions(
        pos_ants, sample_angles, pheromone_maps, seight_radius)

    pheromone_concentrations = get_pheromone_concentrations(sample_positions, pheromone_maps, addition_array_for_pheromone_concentrations)
    best_direction_indices = get_best_direction_indecies(pheromone_concentrations)
    best_direction =  sample_angles[torch.arange(species).unsqueeze(1), torch.arange(ants_per_species), best_direction_indices]

    new_speed =  torch.stack((speed_magnitute *torch.cos(best_direction), speed_magnitute *torch.sin(best_direction)), dim=2)
    
    return new_speed



addition_array_for_pheromone_concentrations = get_addition_array_for_pheromone_concentrations(species, ants_per_species)

frame_count = 0
start_time = time.time()


paused = False
running = True
while running:
    clock.tick(ticks)
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f'FPS: {fps:.2f}')
        frame_count = 0
        start_time = time.time()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not (paused)
    if not paused:
        screen.fill(colors.BLACK)
        
        phero_speed = calculate_new_speed(speed_ants, pos_ants)
        
        speed_ants = speed_ants * (1- pheromone_weight) + phero_speed * pheromone_weight
        speed_ants = speed_ants * (1- randomness_weight) + (torch.rand(size=[species, ants_per_species, 2]).to(device) * 2 - 1) * max_speed * randomness_weight 
        
        pos_ants, speed_ants = handle_collisions(
            pos_ants, speed_ants, ids_pairs, original_shape, HEIGHT, WIDTH, ant_size)

        pheromone_maps = deposit_pheromones(pos_ants, pheromone_maps)
        pheromone_maps = diffuse_pheromones(pheromone_maps)
        pheromone_maps = diffuse_pheromones(pheromone_maps)
        render_pheromones(screen, pheromone_maps, [], color)
        pheromone_maps *= (1-evaporation_factor)

        # ants = pos_ants.reshape(-1, 2)
        # for ant in ants:
        #    pygame.draw.circle(screen, colors.BROWN, ant.tolist(), ant_size)
    pygame.display.update()
pygame.quit()
sys.exit()
