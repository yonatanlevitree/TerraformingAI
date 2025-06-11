import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


class Well:
    def __init__(self, x0, y0, depth, volume):
        self.x0 = x0
        self.y0 = y0
        self.depth = depth
        self.volume = volume

        # Define cost coefficients, they are arbitraty (can be adjusted as needed)
        self.a = 100  # Monetary cost coefficient for depth
        self.b = 50   # Monetary cost coefficient for volume
        self.c = 10   # Time cost coefficient for depth
        self.d = 5    # Time cost coefficient for volume

    def height_at(self, x, y):
        diameter = 3 * self.depth
        radius = diameter / 2
        # Calculate the height of the ellipsoid using the volume formula
        height = (3 * self.volume) / (4 * np.pi * radius ** 2)
        # Apply a Gaussian function to model the influence decay around the well
        return height * torch.exp(-((x - self.x0) ** 2 / (2 * radius ** 2) + (y - self.y0) ** 2 / (2 * radius ** 2)))
    
    def monetary_cost(self):
        # Calculate the monetary cost of the well
        return self.a * self.depth + self.b * self.volume

    def time_cost(self):
        # Calculate the time cost of the well
        return self.c * self.depth + self.d * self.volume


class Terrain:
    def __init__(self, size, noise_level, smoothness, epsilon, device='cpu', regenerate=False):
        self.size = size
        self.noise_level = noise_level
        self.smoothness = smoothness
        self.epsilon = epsilon
        self.device = device
        self.initial_terrain_path = "initial_terrain.npy"
        self.goal_terrain_path = "goal_terrain.npy"
        self.regenerate_terrain(regenerate)

    def regenerate_terrain(self, regenerate):
        if regenerate or not (os.path.exists(self.initial_terrain_path) and os.path.exists(self.goal_terrain_path)):
            self.initial_terrain, self.goal_terrain = self.generate_terrains(
                self.size, self.noise_level, self.smoothness, self.device)
            np.save(self.initial_terrain_path, self.initial_terrain.cpu().numpy())
            np.save(self.goal_terrain_path, self.goal_terrain.cpu().numpy())
        else:
            self.initial_terrain = torch.tensor(np.load(self.initial_terrain_path), device=self.device)
            self.goal_terrain = torch.tensor(np.load(self.goal_terrain_path), device=self.device)

    def generate_terrains(self, size, noise_level, smoothness, device):
        initial_terrain = np.random.rand(size, size) * noise_level
        smoothed_initial_terrain = gaussian_filter(initial_terrain, smoothness)

        additional_height = .25
        goal_terrain = smoothed_initial_terrain + np.abs(
            np.random.rand(size, size) * (noise_level / 2)) + additional_height
        smoothed_goal_terrain = gaussian_filter(goal_terrain, smoothness)

        return (torch.tensor(smoothed_initial_terrain, dtype=torch.float32, device=device),
                torch.tensor(smoothed_goal_terrain, dtype=torch.float32, device=device))
    
    def check_local_fidelity(self, current_terrain, window_size=10):
        """
        Check fidelity using a sliding window of size `window_size x window_size`.
        For every window position, compute the Euclidean norm of the discrepancy
        (difference between current and goal terrains). If any window has a norm
        greater than epsilon, return False. Otherwise True.

        Additionally, return the maximum patch norm encountered for logging.
        """
        size = self.size
        discrepancy = current_terrain - self.goal_terrain

        max_patch_norm = 0.0
        for i in range(size - window_size + 1):
            for j in range(size - window_size + 1):
                patch = discrepancy[i:i+window_size, j:j+window_size]
                patch_norm = torch.norm(patch).item()
                if patch_norm > max_patch_norm:
                    max_patch_norm = patch_norm
                if patch_norm > self.epsilon:
                    # If any window fails, return immediately
                    return False, max_patch_norm
        return True, max_patch_norm

    def scale_terrains(self, fidelity):
        scale_factor = {'low': 0.1, 'medium': .5}.get(fidelity, 1)
        scaled_size = int(self.size * scale_factor)
        self.initial_terrain = torch.nn.functional.interpolate(self.initial_terrain.unsqueeze(0).unsqueeze(0),
                                                               size=(scaled_size, scaled_size),
                                                               mode='bilinear', align_corners=False).squeeze(0).squeeze(
            0)
        self.goal_terrain = torch.nn.functional.interpolate(self.goal_terrain.unsqueeze(0).unsqueeze(0),
                                                            size=(scaled_size, scaled_size),
                                                            mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

    def plot_terrains(self, original_terrain, goal_terrain, optimized_terrain, all_wells, max_points, title):
        plt.figure(figsize=(30, 6))  # Adjusted figure size to accommodate five subplots

        ax1 = plt.subplot(1, 5, 1)
        self.plot_single_terrain(ax1, original_terrain, "Original Terrain")

        ax2 = plt.subplot(1, 5, 2)
        self.plot_single_terrain(ax2, goal_terrain, "Goal Terrain")

        ax3 = plt.subplot(1, 5, 3)
        self.plot_single_terrain(ax3, optimized_terrain, "Optimized Terrain")
        self.annotate_wells_characteristics(ax3, all_wells)

        ax4 = plt.subplot(1, 5, 4)
        self.plot_discrepancy_map(ax4, original_terrain, goal_terrain, "Discrepancy Map")

        # Updated subplot for well effects only
        ax5 = plt.subplot(1, 5, 5)
        self.plot_well_effects(ax5, all_wells, "Cumulative Well Effects")
        self.annotate_wells_characteristics(ax5, all_wells)

        plt.suptitle(title)
        plt.show()

    def plot_well_effects(self, ax, wells, title):
        # Create an empty terrain to accumulate well effects
        effects = torch.zeros((self.size, self.size), device=self.device)
        x = torch.arange(self.size, device=self.device).view(-1, 1).repeat(1, self.size)
        y = torch.arange(self.size, device=self.device).repeat(self.size, 1)
        for well in wells:  # This now includes all wells from all iterations
            effects += well.height_at(x, y)
        im = ax.imshow(effects.cpu().numpy(), cmap='coolwarm', origin='lower', extent=[0, self.size, 0, self.size])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, orientation='vertical')

    def plot_single_terrain(self, ax, terrain, title):
        im = ax.imshow(terrain.cpu().numpy(), cmap='viridis', origin='lower', extent=[0, self.size, 0, self.size])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, orientation='vertical')

    def annotate_wells(self, ax, max_points, label):
        # Plot each well center and add a legend entry
        for point in max_points:
            ax.plot(point[1], point[0], 'ro')  # Red dot for well center
            ax.text(point[1], point[0], f"{label}", color='white', fontsize=8, ha='right')

    def annotate_wells_characteristics(self, ax, wells):
        # Plot each well center and add a legend entry
        for well in wells:
            ax.plot(well.y0, well.x0, 'ro')  # Red dot for well center
            label = f"Depth: {well.depth:.2f}, Volume: {well.volume:.2f}"
            ax.text(well.x0, well.y0, label, color='white', fontsize=8)

    def plot_discrepancy_map(self, ax, original_terrain, goal_terrain, title):
        discrepancy = goal_terrain - original_terrain
        im = ax.imshow(discrepancy.cpu().numpy(), cmap='coolwarm', origin='lower', extent=[0, self.size, 0, self.size])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, orientation='vertical')

    def apply_wells(self, wells):
        # Apply the effects of all wells to the terrain
        x = torch.arange(self.size, device=self.device).view(-1, 1).repeat(1, self.size)
        y = torch.arange(self.size, device=self.device).repeat(self.size, 1)
        new_terrain = self.initial_terrain.clone()
        for well in wells:
            new_terrain += well.height_at(x, y)
        return new_terrain
