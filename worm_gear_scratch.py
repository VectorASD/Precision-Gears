import numpy as np
import meshplot as mp

def create_worm_gear(radius, height, num_teeth, helix_angle):
    # Generate the gear teeth
    num_points = 50
    tooth_width = np.pi * 2 / num_teeth
    teeth = np.zeros((num_points * num_teeth, 3))
    for i in range(num_teeth):
        start_index = i * num_points
        for j in range(num_points):
            angle = j * (2 * np.pi / num_points)
            x = radius * np.cos(angle + i * tooth_width)
            y = radius * np.sin(angle + i * tooth_width)
            z = height / num_points * j
            teeth[start_index + j] = [x, y, z]

    # Generate the worm gear helix
    num_points = 200
    helix = np.zeros((num_points, 3))
    for i in range(num_points):
        angle = i * (2 * np.pi / num_points)
        x = (radius + height / (2 * np.pi) * angle) * np.cos(angle)
        y = (radius + height / (2 * np.pi) * angle) * np.sin(angle)
        z = i / num_points * height
        helix[i] = [x, y, z]

    # Combine the teeth and helix into a single mesh
    mesh = np.concatenate([teeth, helix])

    # Rotate the mesh to create the helix
    rotation = np.array([[np.cos(helix_angle), -np.sin(helix_angle), 0],
                         [np.sin(helix_angle), np.cos(helix_angle), 0],
                         [0, 0, 1]])
    mesh = np.dot(mesh, rotation)

    return mesh

# Create the worm gear mesh
worm_gear = create_worm_gear(radius=10, height=30, num_teeth=12, helix_angle=np.pi / 4)

# Display the worm gear mesh using meshplot
mp.plot(worm_gear[:, 0], worm_gear[:, 1], worm_gear[:, 2], shading={"point_size": 0.05})