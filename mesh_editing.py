import math

import bpy
import bmesh
from bmesh.types import BMesh, BMEdge
from bpy.types import Mesh, Object
from mathutils import Matrix, Vector
from .math_funcs import get_bounding_box, minify_bbox

def boolean_mesh(
    target: Mesh,
    bool_mesh: Mesh,
    operation: str,
    scale: float = 1,
    location: Vector = Vector((0, 0, 0)),
    solver: str = "EXACT",
) -> Mesh:
    valid_operations = ("UNION", "INTERSECT", "DIFFERENCE")
    if operation not in valid_operations:
        err = f"{operation} invalid. Supported operations are: "
        valid_args = str(valid_operations)
        raise ValueError(err + valid_args)

    bool_obj_a = bpy.data.objects.new("__TEMP_BOOL_A", object_data=target)
    bool_obj_b = bpy.data.objects.new("__TEMP_BOOL_A", object_data=bool_mesh)
    bpy.context.scene.collection.objects.link(bool_obj_a)
    bpy.context.scene.collection.objects.link(bool_obj_b)

    # Transform object b
    bool_obj_b.scale = Vector.Fill(3, scale)
    bool_obj_b.location = location

    modifier = bool_obj_a.modifiers.new(name="temp_bool", type="BOOLEAN")
    modifier.solver = solver
    modifier.object = bool_obj_b
    modifier.operation = operation

    # Get resulting mesh
    dg = bpy.context.evaluated_depsgraph_get()
    evaled = bool_obj_a.evaluated_get(dg)
    result = evaled.data.copy()

    # Cleanup
    bpy.data.objects.remove(bool_obj_a)
    bpy.data.objects.remove(bool_obj_b)
    return result


def floor_mesh(mesh: Mesh, offset: float = 0) -> Mesh:
    min_z = minify_bbox(get_bounding_box(mesh))[0].z
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for v in bm.verts:
        v.co.z -= min_z + offset
    bm.to_mesh(mesh)
    return mesh


def translate_mesh(mesh: Mesh, translation: float = 0) -> Mesh:
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for v in bm.verts:
        v.co += translation
    bm.to_mesh(mesh)
    return mesh


def clear_seams(bm: BMesh) -> BMesh:
    for edge in bm.edges:
        edge.seam = False
    return bm


def add_seams_by_angle(bm: BMesh, angle=math.radians(45)) -> BMesh:
    for edge in bm.edges:
        try:
            if edge.calc_face_angle() > angle:
                edge.seam = True
                # print(math.degrees(edge.calc_face_angle()))
        except ValueError:
            continue
    return bm


# mesh = bpy.context.active_object.data
# bm = bmesh.from_edit_mesh(mesh)
# add_seams_by_angle(bm)
# bmesh.update_edit_mesh(mesh)