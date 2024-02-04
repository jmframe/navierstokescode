import numpy as np



def laplacian_3d(phi, dx, dy, dz):
    # Calculate the Laplacian of a 3D scalar field phi
    # phi: 3D numpy array representing the scalar field
    # dx, dy, dz: Grid spacing along x, y, and z axes

    # Calculate second spatial derivatives along each axis
    d2phi_dx2 = np.gradient(np.gradient(phi, axis=0, edge_order=2), axis=0, edge_order=2)
    d2phi_dy2 = np.gradient(np.gradient(phi, axis=1, edge_order=2), axis=1, edge_order=2)
    d2phi_dz2 = np.gradient(np.gradient(phi, axis=2, edge_order=2), axis=2, edge_order=2)

    # Sum the second derivatives to compute Laplacian
    laplacian_phi = d2phi_dx2 / dx**2 + d2phi_dy2 / dy**2 + d2phi_dz2 / dz**2

    return laplacian_phi