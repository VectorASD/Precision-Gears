from itertools import chain
from math import pi, radians
from typing import Deque, List, Tuple

import bpy
import bmesh
from bmesh.types import BMVert, BMFace, BMEdge
from bpy.types import Mesh
from mathutils import Matrix, Vector, Euler

# from . import gear_points


def new_grid_mesh(
    name: str,
    divisions: Tuple[int, int] = (1, 1),
    transform: Matrix = Matrix.Identity(4),
) -> Mesh:
    """
    Create and return a new grid mesh
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm.loops.layers.uv.new("UV")
    bmesh.ops.create_grid(
        bm,
        x_segments=divisions[0],
        y_segments=divisions[1],
        size=1,
        matrix=transform,
        calc_uvs=True,
    )

    bm.to_mesh(mesh)
    bm.free()
    return mesh


def create_set_screw_holes(
    radius: float,
    rotations: Tuple[float],
    resolution: int,
    depth: float,
    z_rot: float = 0,
) -> Mesh:
    mesh = bpy.data.meshes.new("set_screw_holes")
    bm = bmesh.new()

    for rotation in rotations:
        rotation += z_rot

        if bpy.app.version[0] == 2:
            cylinder = bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                cap_tris=False,
                segments=resolution,
                diameter1=radius * 2,
                diameter2=radius * 2,
                depth=depth,
            )
        else:
            cylinder = bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                cap_tris=False,
                segments=resolution,
                radius1=radius,
                radius2=radius,
                depth=depth,
            )

        verts = cylinder["verts"]

        for v in verts:
            v.co.z += depth * 0.5
            v.co.rotate(Euler((pi / 2, 0, 0)))
            v.co.rotate(Euler((0, 0, rotation)))

    bm.to_mesh(mesh)
    return mesh


def new_cube_mesh(size: Vector = Vector((1, 1, 1))):
    mesh = bpy.data.meshes.new("cube")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def new_cross_mesh(
    size: float = 1,
    x_length: float = 4,
    y_length: float = 4,
    rotation: Euler = Euler((0, 0, 0)),
    depth: float = 1,
    center: bool = True,
):
    mesh = bpy.data.meshes.new("cross")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=3, y_segments=3, size=0.5 * 3)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Remove corner points
    to_delete = (0, 3, 12, 15)
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in to_delete])
    bm.verts.ensure_lookup_table()

    x_corners = (2, 6, 5, 9)
    x_corners = [bm.verts[i] for i in x_corners]
    y_corners = (0, 1, 10, 11)
    y_corners = [bm.verts[i] for i in y_corners]

    to_delete = []

    for v in bm.verts:
        v.co *= Vector((size, size, 1))

    if x_length <= size:
        to_delete.extend(x_corners)
    else:
        x_loc = x_length / 2
        x_neg = x_corners[:2]
        x_pos = x_corners[2:]

        for v in x_neg:
            v.co.x = -x_loc
        for v in x_pos:
            v.co.x = x_loc

    if y_length <= size:
        to_delete.extend(y_corners)
    else:
        y_loc = y_length / 2
        y_neg = y_corners[:2]
        y_pos = y_corners[2:]

        for v in y_neg:
            v.co.y = -y_loc
        for v in y_pos:
            v.co.y = y_loc

    bmesh.ops.delete(bm, geom=to_delete)

    if depth != 0:
        extrusion = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
        translate = Vector((0, 0, depth))
        verts = [i for i in extrusion["geom"] if isinstance(i, BMVert)]
        bmesh.ops.translate(bm, vec=translate, verts=verts)

    for v in bm.verts:
        v.co.rotate(rotation)

    if center:
        offset = Vector((0, 0, -depth * 0.5))
        bmesh.ops.translate(bm, vec=offset, verts=bm.verts[:])

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def new_cylinder_mesh(
    radius: float,
    segments: int = 64,
    depth: float = 5,
    rotation: Euler = Euler((0, 0, 0)),
    on_floor: bool = False,
) -> Mesh:
    """
    Create and return a new grid mesh
    """
    mesh = bpy.data.meshes.new("cylinder")
    bm = bmesh.new()
    bm.loops.layers.uv.new("UV")

    if bpy.app.version[0] == 2:
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=True,
            segments=segments,
            diameter1=radius,  # Is this misnamed in api?
            diameter2=radius,
            depth=depth,
        )
    else:
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=True,
            segments=segments,
            radius1=radius,
            radius2=radius,
            depth=depth,
        )

    if on_floor:
        for v in bm.verts:
            v.co.z += (depth * 0.5) - depth * 0.5

    for v in bm.verts:
        v.co.rotate(rotation)

    bm.to_mesh(mesh)
    bm.free()
    return mesh


def new_square_torus_mesh(
    name: str,
    radius_inside: float,
    radius_outside: float,
    segments: int,
    depth: float,
    rotation: Euler = Euler((0, radians(90), 0)),
) -> Mesh:
    """
    TODO: Add bevel
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm.loops.layers.uv.new("UV")

    if bpy.app.version[0] == 2:
        bmesh.ops.create_cone(
            bm,
            cap_ends=False,
            cap_tris=False,
            segments=segments,
            diameter1=radius_inside * 2,
            diameter2=radius_outside * 2,
            depth=0,
        )
    else:
        bmesh.ops.create_cone(
            bm,
            cap_ends=False,
            cap_tris=False,
            segments=segments,
            radius1=radius_inside,
            radius2=radius_outside,
            depth=0,
        )

    extrusion = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    extrusion_points = [
        item for item in extrusion["geom"] if isinstance(item, bmesh.types.BMVert)
    ]

    for p in extrusion_points:
        p.co += Vector((0, 0, depth))

    for v in bm.verts:
        v.co.rotate(rotation)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    bm.to_mesh(mesh)
    bm.free()
    return mesh


def mesh_from_points(*points_iters):
    for index, points in enumerate(points_iters):
        mesh = bpy.data.meshes.new(f"Point cloud {index}")
        bm = bmesh.new()
        for point in points:
            if len(point) == 2:
                point = (*point, 0)
            bm.verts.new(point)
        bm.to_mesh(mesh)
        yield mesh


# def spur_gear(props, mesh: Mesh):
#     points = gear_points.involute_tooth_points(
#         module=props.module,
#         teeth=props.teeth,
#         pressure_angle=props.pressure_angle,
#         clearance=props.clearance,
#         shift=0.5,
#         beta=0.0,
#         undercut=props.undercut,
#         backlash=props.backlash,
#         head=0.00,
#         resolution=20,
#     )

#     pitch_diameter = props.module * props.teeth
#     pitch_radius = pitch_diameter / 2

#     undercut_left = [Vector((point[1], point[0], 0)) for point in points[0]]
#     face_left = [Vector((point[1], point[0], 0)) for point in points[1]]
#     land = [Vector((point[1], point[0], 0)) for point in points[2]]
#     face_right = [Vector((point[1], point[0], 0)) for point in points[3]]
#     undercut_right = [Vector((point[1], point[0], 0)) for point in points[4]]


#     points_left = undercut_left + face_left
#     n_face_side_points = len(points_left)
#     points_right = face_right + undercut_right

#     first_tooth_points = points_left + points_right

#     # Calc front tooth faces
#     n_front_faces = n_face_side_points - 1
#     faces = []
#     for face_idx in range(n_front_faces):
#         face = [
#             0 + face_idx,
#             1 + face_idx,
#             n_face_side_points - 2 - face_idx + n_face_side_points,
#             n_face_side_points - 1 - face_idx + n_face_side_points,
#         ]
#         faces.append(face)

#     bm = bmesh.new()

#     first_tooth_verts = []
#     for idx, point in enumerate(first_tooth_points):
#         vert = bm.verts.new(point)
#         first_tooth_verts.append(vert)

#         if idx == 0 or idx == n_face_side_points * 2 - 1:
#             vert.tag = True

#     bm.verts.ensure_lookup_table()

#     first_tooth_faces = []
#     for face in faces:
#         verts = [bm.verts[i] for i in face]
#         first_tooth_faces.append(bm.faces.new(verts))

#     geom = first_tooth_faces
#     axis = Vector((0, 0, 1))
#     n_copies = props.teeth - 1
#     angle = (2 * pi / props.teeth) * n_copies
#     bmesh.ops.spin(bm, geom=geom, axis=axis, angle=angle, use_duplicate=True, steps=n_copies)

#     bm.verts.ensure_lookup_table()

#     inside_edges = set()
#     inside_verts = [v for v in bm.verts[:] if v.tag]
#     tagged = Deque(inside_verts)

#     while tagged:
#         # start.co += Vector((0, 0, 1))
#         start = tagged.pop()
#         linked_verts = set()
#         for edge in start.link_edges:
#             other_vert = edge.other_vert(start)
#             if other_vert.tag:
#                 inside_edges.add(edge)
#             linked_verts.add(other_vert)

#             continue

#         closest_vert = None
#         closest_vert_dist = None

#         # Get closest unlinked:
#         for v in tagged:
#             if v in linked_verts:
#                 continue

#             dist = (start.co - v.co).length
#             if closest_vert is None or dist < closest_vert_dist:
#                 closest_vert = v
#                 closest_vert_dist = dist

#         end = closest_vert
#         print('toot')
#         inside_edges.add(bm.edges.new((start, end)))

#     bmesh.ops.edgeloop_fill(bm, edges=list(inside_edges))

#     offset_per_setp = props.width / props.z_resolution
#     rot_step = props.helix_angle * (props.width / props.z_resolution)
#     dvec = Vector((0, 0, offset_per_setp))
#     helix_result = bmesh.ops.spin(
#         bm, geom=bm.faces[:],
#         axis=axis,
#         angle=rot_step,
#         dvec=dvec,
#         use_duplicate=False,
#         steps=props.z_resolution,
#     )

#     if props.herringbone:
#         helix_result = helix_result['geom_last']
#         to_delete = []
#         for elem in helix_result:
#             if isinstance(elem, bmesh.types.BMFace):
#                 to_delete.append(elem)

#         bmesh.ops.delete(bm, geom=to_delete, context="FACES")

#     bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

#     bm.to_mesh(mesh)
#     bm.free()
#     return mesh
