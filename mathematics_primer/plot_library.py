import matplotlib.pyplot as plt

def calculate_and_plot_laplacian(laplacian_phi, x, y, z, plot_axis='Z'):
    # Create a 3x3 grid of plots for the Laplacian at different z-values
    fig, axes = plt.subplots(3, 3, figsize=(5, 5))

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if plot_axis == 'Z':
                z_index = i * 3 + j  # Index to choose a z-value
                ax.imshow(laplacian_phi[:, :, z_index], cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'Z = {z[z_index]:.2f}')
            elif plot_axis == 'X':
                x_index = i * 3 + j  # Index to choose an x-value
                ax.imshow(laplacian_phi[x_index, :, :], cmap='viridis', extent=[y.min(), y.max(), z.min(), z.max()])
                ax.set_xlabel('Y')
                ax.set_ylabel('Z')
                ax.set_title(f'X = {x[x_index]:.2f}')
            elif plot_axis == 'Y':
                y_index = i * 3 + j  # Index to choose a y-value
                ax.imshow(laplacian_phi[y_index, :, :], cmap='viridis', extent=[x.min(), x.max(), z.min(), z.max()])
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
                ax.set_title(f'Y = {y[y_index]:.2f}')

    plt.tight_layout()
    plt.show()