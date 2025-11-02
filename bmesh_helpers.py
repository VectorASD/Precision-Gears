# context.area: VIEW_3D
from cmath import polar
from collections import deque
from math import cos, sin, pi, radians
from typing import Dict, Generator, Iterable, Iterator, List, Set, Tuple, Union
from itertools import accumulate, chain

import bpy
import bmesh
from bpy.types import Mesh
from mathutils import Vector
from bmesh.types import BMVert, BMEdge, BMFace, BMesh
import numpy as np

Geom = List[Union[BMVert, BMEdge, BMFace]]


def angle_crease_mesh(mesh: Union[Mesh, BMesh], crease: float) -> None:
    # TODO: Requires 4.0 update re: crease now being a generic attribute
    if isinstance(mesh, Mesh):
        bm = bmesh.new()
        bm.from_mesh(mesh)
    else:
        bm = mesh

    # Ensure crease layer
    if bpy.app.version[0] < 4:
        crease_layer = bm.edges.layers.crease.verify()
    else:
        crease_layer = bm.edges.layers.float.get("crease_edge")
        if not crease_layer:
            crease_layer = bm.edges.layers.float.new("crease_edge")

    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            edge[crease_layer] = 1
            continue
        if edge.calc_face_angle() > crease:
            edge[crease_layer] = 1

    if isinstance(mesh, Mesh):
        bm.to_mesh(mesh)


def polar_sort_verts(verts: Iterable[BMVert], axis: str = 'xy') -> List[BMVert]:
    def _vert_co_as_polar_angle(vert: BMVert) -> float:
        co = getattr(vert.co, axis).to_tuple()
        co = complex(*co)
        angle = polar(co)[1]
        return angle

    return sorted(verts, key=_vert_co_as_polar_angle)


def circle_points(
    npoints: int, radius: float, init_angle: float = 0
) -> Tuple[List[Vector], List[Vector], List[Vector]]:
    """
    Return a circle of vectors with outward normals and tangents
    Clockwise from first position (0, 1, 0)
    Tangent is left handed
    Args:
        npoints: (int): Number of points in circle
        radius: (float): Radius of circle to generate
        init_angle: (float): Offset initial angle in radians

    Returns (positions, normals, tangents)
    """
    positions: List[Vector] = []
    normals: List[Vector] = []
    tangents: List[Vector] = []
    angle_step = (2 * pi) / npoints
    for index in range(npoints):
        angle = init_angle + (angle_step * index)
        position = Vector((sin(angle) * radius, cos(angle) * radius, 0))
        normal = position.normalized()
        tangent = normal.cross(Vector((0, 0, 1)))
        positions.append(position)
        normals.append(normal)
        tangents.append(tangent)

    return positions, normals, tangents


def verts_in_range(verts: List[BMVert], bbox_min: Vector, bbox_max: Vector) -> List[BMVert]:
    """Return verts whose coordinates are within the specified bounding box points

    Args:
        verts (List[BMVert]): List of vertex candidates
        bbox_min (Vector): Minimum bounding box point
        bbox_max (Vector): Maximum bounding box point
    """
    def _vert_in_range(vert):
        def _value_tests():
            yield vert.co.x >= bbox_min.x and vert.co.x <= bbox_max.x
            yield vert.co.y >= bbox_min.y and vert.co.y <= bbox_max.y
            yield vert.co.z >= bbox_min.z and vert.co.z <= bbox_max.z
        return all(_value_tests)
    return map(verts, _vert_in_range)


def shared_faces(verts: Iterable[BMVert]) -> List[BMFace]:
    vert_set = set(verts)
    faces = set()
    all_faces = chain.from_iterable(vert.link_faces for vert in vert_set)
    for face in all_faces:
        if set(face.verts).issubset(vert_set):
            faces.add(face)
    return list(faces)


def shared_edges(verts: Iterable[BMVert]) -> List[BMEdge]:
    vert_set = set(verts)
    edges = set()
    all_edges = chain.from_iterable(vert.link_edges for vert in vert_set)
    for edge in all_edges:
        if set(edge.verts).issubset(vert_set):
            edges.add(edge)
    return list(edges)


def verts_from_faces(faces: Iterable[BMFace]) -> Set[BMVert]:
    """ Return unique vertices used by faces """
    verts = set()
    for face in faces:
        valid_verts = [vert for vert in face.verts if vert.is_valid]
        verts = verts.union(valid_verts)
    return verts


def bisect_geometry(
    bm: BMesh, geom: Geom, dist: float = 0.00001, loc: Vector = Vector((0, 0, 0)), norm: Vector = Vector((0, 0, 1))
) -> Dict[str, Geom]:
    """Bisect Bmesh Wrapper
    TODO: Something wrong with this. Probably the nested function.

    Args:
        bm (BMesh): bmesh
        geom (Geom): geometry to bisect, needs faces, edges, verts
        dist (float): Distance to create new cut
        loc (Vector): Location of cut
        norm (Vector): Cutter normal

    Returns:
        Dict[str, List[Geom]]: 
            "above": Verts above cut.
            "below": Verts below cut.
            "cut": New cut geometry, verts and edges
    """
    def _vert_in_front(vert) -> bool:
        """ Check wether vert position is in front of location and normal"""
        # return norm.dot(vert.co - loc) >= 0
        return norm.dot((vert.co - loc).normalized()) >= 0

    result = bmesh.ops.bisect_plane(bm, geom=geom, dist=dist, plane_co=loc, plane_no=norm)
    # all_geom = result['geom']
    # all_verts = set((i for i in all_geom if isinstance(i, BMVert)))
    # print("all_verts", len(all_verts))
    input_faces = [i for i in geom if isinstance(i, BMFace)]
    input_verts = verts_from_faces(input_faces)
    # print(input_verts)
    cut_geom = result['geom_cut']
    # cut_verts = set((i for i in cut_geom if isinstance(i, BMVert)))
    above_verts = set(filter(_vert_in_front, input_verts))
    # above_verts = above_verts.union(cut_verts)
    below_verts = input_verts - above_verts
    above_verts -= set(cut_geom)
    result = {
        "above": list(above_verts),
        "below": list(below_verts),
        "cut": cut_geom,
    }
    return result


def calc_vertex_normals(
    verts: Iterable[BMVert], face_mask: Union[None, Iterable[BMFace]] = None
) -> List[Vector]:
    """Caclulate vertex normal from link faces with optional face masking

    Args:
        verts (List[BMVert]): verts to calculate normals for 
        face_mask (Union[None, List[BMFace]]): Optional list of faces to limit calculation to 
    """
    if face_mask is not None:
        face_mask = set(face_mask)

    def _calc_normal(vert: BMVert) -> Vector:
        if face_mask is not None:
            link_faces = set(vert.link_faces).intersection(face_mask)
        else:
            link_faces = vert.link_faces
        # print("link_faces", link_faces)
        if len(link_faces) == 0:
            return Vector((0, 0, 0))
        normals = [face.normal for face in link_faces]
        scaler = 1 / len(normals)

        # Average and normalize
        sum_normals = deque(accumulate(normals), maxlen=1).pop()
        average = sum_normals * scaler
        return average.normalized()

    return [_calc_normal(vert) for vert in verts]


def test_func():
    locs, norms, tangents = circle_points(4, 1, init_angle=radians(90))
    for loc, norm, tangent in zip(locs, norms, tangents):
        print(loc, norm, tangent)

# test_func()
