import numpy as np
import copy
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("../data/clothes_models/SafetyVest01.ply")
T = np.eye(4)
T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.deg2rad(90), 0))
T[0, 3] = 1
T[1, 3] = 0
print(T)
mesh_t = copy.deepcopy(mesh).transform(T).scale(0.2, (0,0,0))
o3d.visualization.draw_geometries([mesh, mesh_t])