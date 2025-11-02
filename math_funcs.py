import math
from typing import List, Tuple
from cmath import polar
from functools import singledispatch

import bmesh
from bmesh.types import BMesh
from bpy.types import Mesh, Object
from mathutils import Vector

Points = List[Tuple[float, float, float]]
Faces = List[Tuple[int]]
MeshPlan = Tuple[Points, Faces]


def involute(angle: float):
    """ Involute of a circle. Angle is in radians """
    return math.tan(angle) - angle


def inverse_involute(inv: float):
    """ Return the inverse of the involute """
    b = a
    a = math.atan(0 + inv)

    while abs(a - b) > 0.00001:
        b = a
        a = math.atan(a + inv)
    return a


def sign(n):
    """ Return sign of n"""
    if n > 0:
        return 1
    elif n < 0:
        return -1
    return 0


def polar_to_cartesian(angle, radius):
    return radius * math.cos(angle), radius * math.sin(angle)


def cartesian_to_polar(x, y):
    return polar(complex(x, y))


def mean_vectors(vectors: List[Vector]) -> Vector:
    """ Calculate the mean of the given vectors """
    n_vectors = len(vectors)
    vectors = [v.copy() for v in vectors]
    sum_v = vectors[0]
    for vector in vectors[1:]:
        sum_v += vector

    return sum_v / n_vectors


@singledispatch
def get_bounding_box(_):
    """
    Get the bounding box of the target
    """
    raise NotImplementedError("Unsupported Type")


@get_bounding_box.register(Object)
def _get_object_bbox(target: Object):
    return [Vector(co) for co in target.bound_box]


@get_bounding_box.register(BMesh)
def _get_bmesh_bbox(bm: BMesh):
    try:
        coords = [vert.co for vert in bm.verts]
        # Get X values
        coords.sort(key=lambda v: v.x)
        x_min, x_max = coords[0].x, coords[-1].x
        # Get Y values
        coords.sort(key=lambda v: v.y)
        y_min, y_max = coords[0].y, coords[-1].y
        # Get Z values
        coords.sort(key=lambda v: v.z)
        z_min, z_max = coords[0].z, coords[-1].z

        return [
            Vector((x_min, y_min, z_min)),
            Vector((x_min, y_min, z_max)),
            Vector((x_min, y_max, z_max)),
            Vector((x_min, y_max, z_min)),
            Vector((x_max, y_min, z_min)),
            Vector((x_max, y_min, z_max)),
            Vector((x_max, y_max, z_max)),
            Vector((x_max, y_max, z_min)),
        ]
    except IndexError:
        return [Vector((0, 0, 0))]


@get_bounding_box.register(Mesh)
def _get_mesh_bbox(target: Mesh):
    bm = bmesh.new()
    bm.from_mesh(target)
    bounds = _get_bmesh_bbox(bm)
    bm.free()
    return bounds


def minify_bbox(bbox: List[Vector]) -> Tuple[Vector, Vector]:
    """
    Reduce 8 vector bbox down to 2 vector bbox
    [min, max]
    """
    bbox = list(bbox)
    bbox.sort(key=lambda v: v.x)
    x_min, x_max = bbox[0].x, bbox[-1].x
    # Get Y values
    bbox.sort(key=lambda v: v.y)
    y_min, y_max = bbox[0].y, bbox[-1].y
    # Get Z values
    bbox.sort(key=lambda v: v.z)
    z_min, z_max = bbox[0].z, bbox[-1].z

    return (
        Vector((x_min, y_min, z_min)),
        Vector((x_max, y_max, z_max)),
    )


def map_range(
    value: float, range_a: Tuple[float, float], range_b: Tuple[float, float], clamp01: bool = True
) -> float:
    """Return value in range_a mapped to range_b

    Args:
        value (float): Value to remap
        range_a (Tuple[float, float]): range to map from
        range_b (Tuple[float, float]): range to map to

    Returns:
        float: Remapped value
    """
    (a1, a2), (b1, b2) = range_a, range_b
    remapped = b1 + ((value - a1) * (b2 - b1) / (a2 - a1))
    if clamp01:
        remapped = min(1, max(remapped, 0))
    return remapped
