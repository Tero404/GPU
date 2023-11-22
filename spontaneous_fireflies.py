import torch
import numpy as np
import pygame
from constants import colors


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pygame.init()


WIDTH = 500
HEIGHT = 700
GRAPH_HEIGHT = 200
MAX_SPEED = 1 # maximum speed at a direction of x or y

'''
Warning high speeds or a small grid can lead to visual bugs
'''

FIREFLYS = 150 #number of fireflies

max_energy = 500 #energy at with the firefly glows
overload = 20 #engerylevel above max energy, at that level glowing stops and resets

nudge = 105 #energy given from nearby fireflies
update_distance = 70 #distance at with fireflies interact
ff_clock = 1 #clock speed of fireflies, one ff_tick per frame


FONT = pygame.font.SysFont("Arial", 16)


graph_scale = GRAPH_HEIGHT / (max_energy+overload)


def get_staring_values(width, height, max_speed, particles):
    coords_x = torch.randint(low=0, high=width, size=(particles,)).to(device)
    coords_y = torch.randint(low=0, high=height, size=(particles,)).to(device)
    coords = torch.stack((coords_x, coords_y), dim=1).float()
    speed = (torch.randn((particles, 2)).to(device) - 0.5) * max_speed * 2
    charge = torch.randint(low=0, high=(max_energy+1),
                           size=(particles,)).to(device).float()
    return coords, speed, charge


def update_position(positon, speed, width, height):
    positon += speed
    positon %= torch.tensor([width, height], device=device)
    return positon, speed


def update_clock(charge):
    charge += ff_clock
    return charge


def get_distances(position):
    
    x_pos = position.T[0]
    y_pos = position.T[1]
    
    
    x_dist = torch.abs(x_pos.unsqueeze(0) - x_pos.unsqueeze(1))
    y_dist = torch.abs(y_pos.unsqueeze(0) - y_pos.unsqueeze(1))
    
    x_dist = torch.minimum(x_dist, WIDTH - x_dist)
    y_dist = torch.minimum(y_dist, WIDTH - GRAPH_HEIGHT - y_dist)

    dist = torch.sqrt(x_dist**2 + y_dist**2)
    
    return dist


def fill_diags(tensor: torch.Tensor, fill_Value: float) -> torch.Tensor:
    diag_indices = torch.arange(min(tensor.shape)).to(tensor.device)
    tensor[diag_indices, diag_indices] = fill_Value
    return tensor


def nudge_clock(charges, positons):
    #update_distance = 10
    #nudge = 2

    dist = get_distances(positons)
    max_dist = torch.zeros_like(dist)
    charged = torch.zeros_like(charges)

    max_dist = torch.where(dist <= update_distance,
                           torch.tensor(1.0), torch.tensor(0.0))
    
    max_dist = fill_diags(max_dist, 0.0)

    charged = torch.where(charges >= max_energy,
                         torch.tensor(1.0), torch.tensor(0.0))
    not_charged = torch.where(charges >= max_energy,
                             torch.tensor(0.0), torch.tensor(1.0))

    add_charge = torch.mv(max_dist, charged)
    add_charge = torch.where(add_charge > 0,
                         torch.tensor(1.0), torch.tensor(0.0))
    add_charge = add_charge * nudge * not_charged
    #print (add_charge.shape)


    return charges + add_charge


def plot_to_surface(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    size = fig.canvas.get_width_height()
    return pygame.image.fromstring(buf, size, "RGB")




def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Firefly Simulaton")
    width, height = WIDTH, HEIGHT - GRAPH_HEIGHT

    coords, speed, charge = get_staring_values(
        width, height, MAX_SPEED, FIREFLYS)

    averages = []
    brights = []

    running = True
    clock = pygame.time.Clock()
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(colors.BLACK)

        coords, speed = update_position(coords, speed, width, height)
        particle_positions = coords.tolist()


        bright = 0
        for i, particle in enumerate(particle_positions):
            if charge[i] < max_energy:
                pygame.draw.circle(screen, colors.BROWN, particle, 3)

            elif charge[i] >= max_energy + overload:
                pygame.draw.circle(screen, colors.YELLOW, particle, 4)
                charge[i] = -1
                bright += 1

            else:
                pygame.draw.circle(screen, colors.YELLOW, particle, 4)
                bright += 1

        charge = nudge_clock(charge, coords)
        charge = update_clock(charge)
        
        brights.append(bright)
        
        average = float(torch.mean(charge).to('cpu').item())
        averages.append(average)
        
        if len(averages) >3:
            firefly_txt = FONT.render(f"Active Fireflies", 5, colors.YELLOW)
            firefly_avg_txt = FONT.render(f"Average energy", 5, colors.BROWN)
            screen.blit(firefly_txt,(0, HEIGHT - GRAPH_HEIGHT))
            screen.blit(firefly_avg_txt,(WIDTH - firefly_avg_txt.get_width(), HEIGHT - GRAPH_HEIGHT))
            
            averages_coords_y = -np.array(averages) * (GRAPH_HEIGHT - firefly_txt.get_height()) / (max_energy+overload) + HEIGHT 
            brights_coords_y = - np.array(brights) * (GRAPH_HEIGHT - firefly_txt.get_height()) / FIREFLYS + HEIGHT
            
            averages_coords_x = np.arange(len(averages)) * (WIDTH/len(averages))
            
            avg_coords = np.stack((averages_coords_x,averages_coords_y)).T
            brg_coords = np.stack((averages_coords_x,brights_coords_y)).T
            
            
            
            #pygame.draw.rect(screen, colors.WHITE, pygame.Rect(0, HEIGHT-GRAPH_HEIGHT, WIDTH, GRAPH_HEIGHT))
            pygame.draw.lines(screen, colors.BROWN, False, avg_coords, 1)
            pygame.draw.lines(screen, colors.YELLOW, False, brg_coords, 1)
        
        if len(averages) > WIDTH:
            averages.pop(0)
            brights.pop(0)
        

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
