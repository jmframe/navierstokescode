import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_fields(u, v, w, p, z_index, title=False, colorbar=False):
    """
    Plot the velocity (u, v, w) and pressure (p) fields in a row of subplots.

    Args:
        u (torch.Tensor): Velocity field in the x-direction.
        v (torch.Tensor): Velocity field in the y-direction.
        w (torch.Tensor): Velocity field in the z-direction.
        p (torch.Tensor): Pressure field.
        z_index (int): The z-index for visualization.

    Returns:
        None
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(8, 3))

    # Visualize the u field
    im = axes[0].imshow(u[:, :, z_index], cmap='viridis')
    if title:
        axes[0].set_title('u Field')
    axes[0].set_aspect('equal')
    axes[0].grid(False)
    if colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('u')

    # Visualize the v field
    im = axes[1].imshow(v[:, :, z_index], cmap='viridis')
    if title:
        axes[1].set_title('v Field')
    axes[1].set_aspect('equal')
    axes[1].grid(False)
    if colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('v')

    # Visualize the w field
    im = axes[2].imshow(w[:, :, z_index], cmap='viridis')
    if title:
        axes[2].set_title('w Field')
    axes[2].set_aspect('equal')
    axes[2].grid(False)
    if colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('w')

    # Visualize the pressure field
    im = axes[3].imshow(p[:, :, z_index], cmap='viridis')
    if title:
        axes[3].set_title('Pressure Field')
    axes[3].set_aspect('equal')
    axes[3].grid(False)
    if colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Pressure')

    plt.tight_layout()
    plt.show()
    plt.close()


# Function to set up initial pressure with a single high-pressure point
def initialize_pressure_with_high_point_3d(nx, ny, nz, high_pressure_location):
    hp_x, hp_y, hp_z = high_pressure_location
    # Initialize velocity field (u, v, w)
    u = torch.zeros((nx, ny, nz))
    v = torch.zeros((nx, ny, nz))
    w = torch.zeros((nx, ny, nz))

    # Specify the location of the high-pressure point (e.g., at the center)
    high_pressure_location = (nx // hp_x, ny // hp_y, nz // hp_z)
    p_initial = torch.zeros((nx, ny, nz))
    p_initial[high_pressure_location] = 10.0  # Set a high pressure value at the specified location
    return p_initial, u, v, w

def solve_continuity_equation_3d(u, v, w, p, dx, dy, dz, dt, num_iterations):
    # Get the dimensions from the input tensors
    nx, ny, nz = u.shape

    # Define fluid properties
    rho = 1.0  # Density (assumed constant for incompressible flow)
    mu = 0.1  # Dynamic viscosity

    for n in range(num_iterations):
        # Calculate divergence of velocity field
        div_u = torch.zeros((nx, ny, nz))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    div_u[i, j, k] = (
                        (u[i + 1, j, k] - u[i - 1, j, k]) / (2 * dx)
                        + (v[i, j + 1, k] - v[i, j - 1, k]) / (2 * dy)
                        + (w[i, j, k + 1] - w[i, j, k - 1]) / (2 * dz)
                    )

        # Update pressure using Poisson equation (Laplacian of pressure)
        for _ in range(20):  # Iterative solver for pressure (adjust as needed)
            p_old = p.clone()
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        p[i, j, k] = (
                            (rho / dt) * div_u[i, j, k]
                            + (
                                mu
                                * (
                                    (p[i + 1, j, k] + p[i - 1, j, k]) / (dx ** 2)
                                    + (p[i, j + 1, k] + p[i, j - 1, k]) / (dy ** 2)
                                    + (p[i, j, k + 1] + p[i, j, k - 1]) / (dz ** 2)
                                )
                            )
                        ) / (
                            (2 * mu / (dx ** 2))
                            + (2 * mu / (dy ** 2))
                            + (2 * mu / (dz ** 2))
                            + (rho / dt)
                        )

        # Update velocity fields using pressure gradients
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    u[i, j, k] -= (dt / dx) * (p[i + 1, j, k] - p[i, j, k]) / rho
                    v[i, j, k] -= (dt / dy) * (p[i, j + 1, k] - p[i, j, k]) / rho
                    w[i, j, k] -= (dt / dz) * (p[i, j, k + 1] - p[i, j, k]) / rho

    return u, v, w, p

def calculate_rate_of_strain_2D(u, v):
    # Using PyTorch to compute gradients
    du_dx = torch.gradient(u, spacing=(1.0,), dim=1)[0]  # Gradient with respect to x
    dv_dy = torch.gradient(v, spacing=(1.0,), dim=0)[0]  # Gradient with respect to y
    du_dy = torch.gradient(u, spacing=(1.0,), dim=0)[0]  # Gradient with respect to y
    dv_dx = torch.gradient(v, spacing=(1.0,), dim=1)[0]  # Gradient with respect to x

    # Combining the tensors correctly
    rate_of_strain = torch.stack([
        torch.stack([du_dx, (du_dy + dv_dx)/2], dim=2),
        torch.stack([(du_dy + dv_dx)/2, dv_dy], dim=2)
    ], dim=3)

    return rate_of_strain

def calculate_rate_of_strain_3D(u, v, w):
    # Calculate the gradient for each component along each axis
    du_dx = torch.gradient(u, dim=2)
    dv_dy = torch.gradient(v, dim=1)
    dw_dz = torch.gradient(w, dim=0)

    du_dy = torch.gradient(u, dim=1)
    du_dz = torch.gradient(u, dim=0)

    dv_dx = torch.gradient(v, dim=2)
    dv_dz = torch.gradient(v, dim=0)

    dw_dx = torch.gradient(w, dim=2)
    dw_dy = torch.gradient(w, dim=1)

    # Construct the rate of strain tensor
    rate_of_strain = torch.tensor([[[du_dx, (du_dy + dv_dx)/2, (du_dz + dw_dx)/2],
                                    [(du_dy + dv_dx)/2, dv_dy, (dv_dz + dw_dy)/2],
                                    [(du_dz + dw_dx)/2, (dv_dz + dw_dy)/2, dw_dz]]])
    return rate_of_strain

# Function to calculate the deviatoric stress tensor
def calculate_deviatoric_stress(viscosity, rate_of_strain):
    # For a Newtonian fluid, τ = μ * (rate of strain tensor)
    return viscosity * rate_of_strain