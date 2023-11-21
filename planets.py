import numpy as np
import torch
import math
import pygame
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


pygame.init()

AU = 149.6e6 * 1000
G = 6.67428e-11
WIDTH, HEIGHT = 1000, 1000
SCALE = 50 / AU  # AU ... Astonomical unit  # 1AU = 100 pixels
TIMESTEP = 3600 * 24
TICKS = 60 * 2000


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet Simulaton")


YELLOW = (255, 255, 0)
DARK_GREY = (169, 169, 169)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
GOLD = (255, 215, 0)
CYAN = (0, 255, 255)
DARK_BLUE = (0, 0, 139)
BLACK = (0, 0, 0)

FONT = pygame.font.SysFont("Arial", 16)


class Planet:

    def __init__(self, name, x, y, radius, color, sun):
        self.name = name
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

        self.orbit = []
        self.sun = sun
        self.distance_to_sun = 0

    def draw(self, screen):
        if len(self.orbit) > 1:
            pygame.draw.lines(surface=screen, color=self.color,
                              closed=False, points=self.orbit, width=2)

        x = self.x * SCALE + WIDTH / 2
        y = self.y * SCALE + HEIGHT / 2
        pygame.draw.circle(screen, self.color, (x, y), self.radius)

        self.orbit.append((x, y))

        # if not self.sun:
        #    distance_txt = FONT.render(f"{round(self.distance_to_sun/1000, 1)}km",1, WHITE)
        #    screen.blit(distance_txt,(x - distance_txt.get_width()/2,y - distance_txt.get_height()/2))


def fill_diags(tensor: torch.Tensor, fill_Value: float) -> torch.Tensor:
    diag_indices = torch.arange(min(tensor.shape)).to(tensor.device)
    tensor[diag_indices, diag_indices] = fill_Value
    return tensor


def get_distances(x_coords: torch.Tensor, y_coords: torch.Tensor) -> torch.Tensor:
    epsilon = 1.0
    x_coords_diff = x_coords.unsqueeze(0) - x_coords.unsqueeze(1)
    y_coords_diff = y_coords.unsqueeze(0) - y_coords.unsqueeze(1)
    distance = torch.sqrt(x_coords_diff**2 + y_coords_diff**2)
    distance = fill_diags(distance, epsilon)
    return distance, x_coords_diff, y_coords_diff


def update_positions(coords: torch.Tensor, masses: torch.Tensor, velocities: torch.Tensor, G: np.float64, TIMESTEP: int) -> torch.Tensor:
    x_coords = coords.T[0]
    y_coords = coords.T[1]

    masses = masses

    distance, x_coords_diff, y_coords_diff = get_distances(x_coords, y_coords)

    forces = G * torch.outer(masses, masses) / distance**2
    forces = fill_diags(forces, 0.0)

    thetas = torch.arctan2(y_coords_diff, x_coords_diff)
    forces_x = (torch.cos(thetas) * forces).sum(axis=1)
    forces_y = (torch.sin(thetas) * forces).sum(axis=1)

    velocities.T[0] += forces_x * TIMESTEP / masses
    velocities.T[1] += forces_y * TIMESTEP / masses

    coords.T[0] += velocities.T[0] * TIMESTEP
    coords.T[1] += velocities.T[1] * TIMESTEP

    distance_to_sun = distance[0]

    return coords, velocities, distance_to_sun


def main():
    run = True
    clock = pygame.time.Clock()

    planets_df = pd.DataFrame([
        ['sun',     0.000 * AU, 0, 3, YELLOW,
            1.988922 * 10**30, 0.0,  0.000 * 1000, True],
        ['mercury', 0.387 * AU, 0, 3, DARK_GREY,
            3.300000 * 10**23, 0.0, -47.400 * 1000, False],
        ['venus',   0.723 * AU, 0, 3, WHITE,
            4.8685 * 10**24, 0.0, -35.020 * 1000, False],
        ['earth',   -1.000 * AU, 0, 3, BLUE,
            5.974200 * 10**24, 0.0,  29.783 * 1000, False],
        ['mars',    -1.524 * AU, 0, 3, RED,
            6.390000 * 10**23, 0.0,  24.077 * 1000, False],
        ['jupiter', -5.203 * AU, 0, 3, ORANGE,
            1.8982 * 10**27, 0.0,  13.070 * 1000, False],
        # ['saturn',  9.537 * AU, 0, 3, GOLD,       5.6834   * 10**26, 0.0,   9.690 * 1000, False],
        # ['uranus', 19.191 * AU, 0, 3, CYAN,       8.6810   * 10**25, 0.0,   6.810 * 1000, False],
        # ['neptune',30.069 * AU, 0, 3, DARK_BLUE,  1.02413  * 10**26, 0.0,   5.430 * 1000, False]
    ], columns=['body', 'x', 'y', 'size', 'color', 'mass', 'x_velocity', 'y_velocity', 'is_sun'])

    planets = torch.from_numpy(
        planets_df[['x', 'y', 'mass', 'x_velocity', 'y_velocity']].to_numpy()).to(device)
    coords = planets.T[:2].T
    masses = planets.T[2]
    velocities = planets.T[3:5].T

    planet_objects = []
    for index, row in planets_df.iterrows():
        planet_obj = Planet(row['body'], row['x'], row['y'],
                            row['size'], row['color'], row['is_sun'])
        planet_objects.append(planet_obj)

    while run:
        clock.tick(TICKS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        screen.fill(BLACK)

        coords, velocities, distance_to_sun = update_positions(
            coords, masses, velocities, G, TIMESTEP)
        coords2 = coords - coords[5]

        for i, planet in enumerate(planet_objects):
            planet.x = coords2[i][0].item()
            planet.y = coords2[i][1].item()
            # planet.distance_to_sun = distance_to_sun[i].item()
            planet.draw(screen)

        pygame.display.update()

    pygame.quit()


main()
