from typing import Tuple

import bpy
import bmesh
from bpy.types import Mesh, Object
from mathutils import Matrix, Vector

from . import mesh_generation
from .mesh_editing import boolean_mesh



# def cut_bore(gear: Mesh, radius: float, resolution: int = 64):
#     bore = mesh_generation.new_cylinder_mesh(
#         radius, resolution, depth=1000,
#     )

#     result = boolean_mesh(gear, bore, "DIFFERENCE")
#     return result


# def add_recess(gear: Mesh, radius: float, resolution: int = 64):
#     bore = mesh_generation.new_cylinder_mesh(
#         "bore", radius, resolution, depth=1000,
#     )

#     result = boolean_mesh(gear, bore, "DIFFERENCE")
#     return result
