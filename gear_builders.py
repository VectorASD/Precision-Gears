from __future__ import division, annotations
from collections import deque
from dataclasses import dataclass
from functools import cached_property
import itertools

from math import isnan, radians, sin, cos, tan, pi, ceil, tau
from typing import TYPE_CHECKING, Deque, List, Union

import bpy
import bmesh
from bpy.types import Mesh, PropertyGroup
import numpy as np
from mathutils import Quaternion, Vector, Euler, Matrix
from mathutils.geometry import interpolate_bezier
from bmesh.types import BMEdge, BMFace, BMLayerItem, BMVert, BMesh
from itertools import chain

from .bmesh_filters import (
    dict_by_type,
    unique_face_edges,
    unique_face_verts,
    unique_vert_edges,
    unique_vert_faces,
    verts_by_layer,
)
from .math_funcs import map_range
from .mesh_editing import boolean_mesh
from .bmesh_helpers import (
    shared_faces,
    shared_edges,
    bisect_geometry,
    calc_vertex_normals,
    polar_sort_verts,
    circle_points,
    verts_from_faces,
)

from .pygears.involute_tooth import InvoluteTooth, InvoluteRack

if TYPE_CHECKING:
    from .props import PrecisionGearsProps


# def spherical_rot(point, phi):
#     new_phi = np.sqrt(np.linalg.norm(point)) * phi
#     return rotation3D(new_phi)(point)


# def make_circle(
#     radius: float, start: float, end: float, samples: int
# ) -> List[np.ndarray]:
#     circle = []
#     angles = np.linspace(start, end, samples)

#     # for angle in np.arange(start, end, angle_range):
#     for angle in angles:
#         circle.append((cos(angle) * radius, sin(angle) * radius, 0))
#     return circle


@dataclass
class GearBuilder:
    type: Union[str, None]
    props: PropertyGroup
    # Overwrite with BMLayerItem references once they're created
    flank_a_vert_grp: Union[str, BMLayerItem] = "flank_a"
    flank_b_vert_grp: Union[str, BMLayerItem] = "flank_b"
    tooth_inside_vert_grp: Union[str, BMLayerItem] = "tooth_inside"
    tooth_tips_vert_grp: Union[str, BMLayerItem] = "tooth_tips"
    tooth_root_vert_grp: Union[str, BMLayerItem] = "tooth_root"
    tooth_index_vert_grp: Union[str, BMLayerItem] = "tooth_index"  # 0 = Not a tooth
    top_verts_grp: Union[str, BMLayerItem] = "top_verts"  # 0 = Not a tooth
    bottom_verts_grp: Union[str, BMLayerItem] = "bottom_verts"  # 0 = Not a tooth
    module_vert_grp: Union[str, BMLayerItem] = "module_vert_grp"
    tooth_id_face_grp: Union[str, BMLayerItem] = "tooth_id"  # 0 = Not a tooth

    def initialize_bmesh_layers(self, bm: BMesh):
        # This should be procedural
        self.flank_a_vert_grp = bm.verts.layers.int.new(self.flank_a_vert_grp)
        self.flank_b_vert_grp = bm.verts.layers.int.new(self.flank_b_vert_grp)
        self.tooth_inside_vert_grp = bm.verts.layers.int.new(self.tooth_inside_vert_grp)
        self.tooth_tips_vert_grp = bm.verts.layers.int.new(self.tooth_tips_vert_grp)
        self.tooth_root_vert_grp = bm.verts.layers.int.new(self.tooth_root_vert_grp)
        self.tooth_index_vert_grp = bm.verts.layers.int.new(self.tooth_index_vert_grp)
        self.module_vert_grp = bm.verts.layers.int.new(self.module_vert_grp)
        self.tooth_id_face_grp = bm.faces.layers.int.new(self.tooth_id_face_grp)
        self.top_verts_grp = bm.verts.layers.int.new(self.top_verts_grp)
        self.bottom_verts_grp = bm.verts.layers.int.new(self.bottom_verts_grp)

    @property
    def pitch_circle_radius(self) -> float:
        radius = (self.props.teeth * self.props.module) / 2
        radius += self.props.module * self.props.shift
        return radius

    def add_flank_end_relief(self, bm):
        if self.props.flank_mod_param_a == 0 or self.props.flank_mod_param_b == 0:
            return None
        flank_a_vert_grp = self.flank_a_vert_grp
        flank_b_vert_grp = self.flank_b_vert_grp

        amnt = self.props.flank_mod_param_a
        dist = self.props.flank_mod_param_b
        height = self.props.height

        cut_positions = (Vector((0, 0, height - dist)), Vector((0, 0, dist)))
        cut_normals = (Vector((0, 0, 1)), Vector((0, 0, -1)))
        remap_range = (
            (height - dist, height),
            (dist, 0),
        )

        # Imposssible parameters
        for cut_p, cut_n, range_a in zip(cut_positions, cut_normals, remap_range):
            dist = min(dist, height * 0.5)

            geom = bm.faces[:] + bm.edges[:] + bm.verts[:]
            result_top = bisect_geometry(bm, geom, loc=cut_p, norm=cut_n)
            top_verts = set(
                [elem for elem in result_top["above"] if isinstance(elem, BMVert)]
            )
            cut_verts = set(
                [elem for elem in result_top["cut"] if isinstance(elem, BMVert)]
            )

            flank_a = set([vert for vert in bm.verts if vert[flank_a_vert_grp] == 1])
            flank_b = set([vert for vert in bm.verts if vert[flank_b_vert_grp] == 1])
            flanks = flank_a.union(flank_b)

            relief_verts = top_verts.intersection(flanks)
            relief_a_verts = relief_verts.intersection(flank_a)
            relief_b_verts = relief_verts.intersection(flank_b)

            relief_a_faces = shared_faces(relief_a_verts.union(cut_verts))
            relief_a_normals = calc_vertex_normals(relief_a_verts, relief_a_faces)
            relief_b_faces = shared_faces(relief_b_verts.union(cut_verts))
            relief_b_normals = calc_vertex_normals(relief_b_verts, relief_b_faces)

            verts = chain(relief_a_verts, relief_b_verts)
            normals = chain(relief_a_normals, relief_b_normals)
            normal_flattener = Vector((1, 1, 0))
            normals = [(n * normal_flattener).normalized() for n in normals]

            for vert, normal in zip(verts, normals):
                init_z = vert.co.z
                mag_scaler = map_range(init_z, range_a, (0, 1))
                mag = amnt * mag_scaler
                vert.co += normal * -mag

    def add_flank_crowing(self, bm):
        if self.props.flank_mod_param_a == 0:
            return None

        flank_a_verts = set(
            [vert for vert in bm.verts if vert[self.flank_a_vert_grp] == 1]
        )
        flank_a_faces = shared_faces(flank_a_verts)
        flank_a_normals = calc_vertex_normals(flank_a_verts, flank_a_faces)

        flank_b_verts = set(
            [vert for vert in bm.verts if vert[self.flank_b_vert_grp] == 1]
        )
        flank_b_faces = shared_faces(flank_b_verts)
        flank_b_normals = calc_vertex_normals(flank_b_verts, flank_b_faces)
        height = self.props.height

        amnt = self.props.flank_mod_param_a
        bias = self.props.flank_mod_param_b
        bias = min(bias, height * 0.5)
        bezier_res = self.props.z_resolution * 4

        ref_bezier = interpolate_bezier(
            Vector((amnt, 0, 0)),
            Vector((0, 0, bias)),
            Vector((0, 0, height - bias)),
            Vector((amnt, 0, height)),
            bezier_res,
        )

        interp_x = np.linspace(0, height, bezier_res)
        interp_y = [v.x for v in ref_bezier]

        verts = list(chain(flank_a_verts, flank_b_verts))
        height_values = [v.co.z for v in verts]
        normals = chain(flank_a_normals, flank_b_normals)
        normal_flattener = Vector((1, 1, 0))
        normals = [(n * normal_flattener).normalized() for n in normals]
        mag_scalers = np.interp(height_values, interp_x, interp_y)

        for vert, normal, mag_scale in zip(verts, normals, mag_scalers):
            vert.co += -normal * mag_scale

    def shade_smooth(self, bm):
        for f in bm.faces:
            f.smooth = True


class InvoluteGear(GearBuilder):
    type = "SPUR"

    def __init__(self, props: PrecisionGearsProps):
        self.props = props
        self.involute_tooth = InvoluteTooth(
            m=props.module,
            z=props.teeth,
            pressure_angle=props.pressure_angle,
            clearance=props.clearance,
            shift=props.shift,
            beta=props.helix_angle,
            undercut=props.undercut,
            backlash=props.backlash,
            head=props.head,
            properties_from_tool=False,
        )

    def create_gear(self) -> Mesh:
        bm = bmesh.new()
        self.initialize_bmesh_layers(bm)
        mesh = bpy.data.meshes.new(self.type)
        is_internal = self.props.gear_type == "INTERNAL"

        height = self.props.height
        helix_angle = self.props.helix_angle
        z_resolution = self.props.z_resolution
        n_teeth = self.props.teeth
        module = self.props.module
        pitch_circle_radius = self.pitch_circle_radius
        tip_radius = pitch_circle_radius + self.props.module

        pts = self.involute_tooth.points(num=self.props.complexity)

        # Check for nan in points
        for co in itertools.chain.from_iterable(pts):
            for value in co:
                if isnan(value):
                    print(co, "Impossible value detected")
                    return None

        if len(pts) == 5:
            side_a_pts = list(chain(*pts[0:2]))
            side_b_pts = list(chain(*pts[3:]))
            side_b_pts.reverse()
        elif len(pts) == 3:
            side_a_pts = list(chain(pts[0]))
            side_b_pts = list(chain(pts[2]))
            side_b_pts.reverse()
        else:
            print(f"Unrecognized involute tooth point set count {len(pts)}")
            return None

        # Mesh first tooth
        ref_tooth_verts: List[BMVert] = []
        ref_tooth_faces: List[BMFace] = []
        for side_a, side_b in zip(side_a_pts, side_b_pts):
            new_vert_a = bm.verts.new(Vector(side_a).to_3d())
            new_vert_b = bm.verts.new(Vector(side_b).to_3d())

            if new_vert_a.co.length >= pitch_circle_radius:
                new_vert_a[self.module_vert_grp] = 1
                new_vert_b[self.module_vert_grp] = 1

            if is_internal:
                new_vert_a.co.x += -(pitch_circle_radius * 2)
                new_vert_b.co.x += -(pitch_circle_radius * 2)

            new_vert_a[self.flank_a_vert_grp] = 1
            new_vert_b[self.flank_b_vert_grp] = 1
            if ref_tooth_verts:
                face = bm.faces.new([*ref_tooth_verts[-2:], new_vert_b, new_vert_a])
                face[self.tooth_id_face_grp] = 1
                face.normal_flip()
                ref_tooth_faces.append(face)

            ref_tooth_verts.append(new_vert_a)
            ref_tooth_verts.append(new_vert_b)

        # Identify tooth features
        ref_tooth_tip_verts = ref_tooth_verts[-2:]
        for vert in ref_tooth_tip_verts:
            vert[self.tooth_tips_vert_grp] = 1

        # Identify ref inside verts
        ref_tooth_inside_verts = ref_tooth_verts[:2]

        for vert in ref_tooth_inside_verts:
            vert[self.tooth_inside_vert_grp] = 1

        # Assign index to ref tooth
        for vert in ref_tooth_verts:
            vert[self.tooth_index_vert_grp] = 1

        # Create teeth from reference tooth
        axis = Vector((0, 0, 1))
        step_angle = 2 * pi / n_teeth
        for tooth_id in range(1, n_teeth):
            angle = step_angle * tooth_id
            result = bmesh.ops.spin(
                bm,
                geom=ref_tooth_faces,
                angle=angle,
                axis=axis,
                steps=1,
                use_duplicate=True,
            )
            type_sorted_result = dict_by_type(result["geom_last"])
            for face in type_sorted_result[BMFace]:
                face[self.tooth_id_face_grp] = tooth_id + 1
            for vert in type_sorted_result[BMVert]:
                vert[self.tooth_index_vert_grp] = tooth_id + 1

        inside_verts = [
            vert for vert in bm.verts if vert[self.tooth_inside_vert_grp] == 1
        ]

        # Create inside Face
        inside_verts = polar_sort_verts(inside_verts)
        center_face = bm.faces.new(inside_verts)
        center_face[self.tooth_id_face_grp] = 0

        # Create outside perimiter
        if is_internal:
            center_face_edges = center_face.edges[:]
            bmesh.ops.delete(bm, geom=[center_face], context="FACES_ONLY")
            outside_ring = dict_by_type(
                bmesh.ops.extrude_edge_only(bm, edges=center_face_edges)["geom"]
            )
            for face in outside_ring[BMFace]:
                face[self.tooth_id_face_grp] = 0
            outside_verts = outside_ring[BMVert]
            outside_verts = polar_sort_verts(outside_verts)
            outside_radius = tip_radius + self.props.width
            ref_angle_offset = -pi * 0.5 + (2 * pi / (n_teeth * 2) / 2)
            ref_circle = list(
                circle_points(len(outside_verts), outside_radius, ref_angle_offset)[0]
            )
            ref_circle.reverse()

            for v, ref in zip(outside_verts, ref_circle):
                v.co.normalize()
                v.co = ref

            rot_offset = Euler((0, 0, pi))
            for v in bm.verts:
                v.co.rotate(rot_offset)

        else:
            poked = bmesh.ops.poke(bm, faces=[center_face])
            for face in poked["faces"]:
                face[self.tooth_id_face_grp] = 0

        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0001)

        if self.props.tooth_mod_enabled:
            if self.props.tip_relief != 0:
                amount = self.props.tip_relief
                segments = self.props.tip_relief_resolution
                tooth_tip_verts = [
                    vert for vert in bm.verts if vert[self.tooth_tips_vert_grp] == 1
                ]
                bmesh.ops.bevel(
                    bm,
                    geom=tooth_tip_verts,
                    offset=amount,
                    segments=segments,
                    profile=0.5,
                )

        if height == 0:
            self.shade_smooth(bm)
            bm.to_mesh(mesh)
            return mesh

        elif helix_angle == 0:
            offset_per_step = height / z_resolution
            extrude_pos = Vector((0, 0, offset_per_step))
            axis = Vector((0, 0, 1))
            bmesh.ops.spin(
                bm, geom=bm.faces, dvec=extrude_pos, axis=axis, steps=z_resolution
            )

        else:  # Has angle
            offset_per_step = height / z_resolution
            axis = Vector((0, 0, 1))
            rotation = ((2 * pi) * height * tan(helix_angle)) / (module * n_teeth * pi)
            dvec = Vector((0, 0, offset_per_step))

            if self.props.herringbone:
                rotation *= 0.5
                dvec *= 0.5

            helix_result = bmesh.ops.spin(
                bm,
                geom=bm.faces[:],
                axis=axis,
                angle=rotation,
                dvec=dvec,
                use_duplicate=False,
                steps=self.props.z_resolution,
            )

            if self.props.herringbone:
                spin_source = [
                    f for f in helix_result["geom_last"] if isinstance(f, BMFace)
                ]
                helix_result = bmesh.ops.spin(
                    bm,
                    geom=spin_source,
                    axis=axis,
                    angle=-rotation,
                    dvec=dvec,
                    use_duplicate=False,
                    steps=self.props.z_resolution,
                )
                bmesh.ops.delete(bm, geom=spin_source, context="FACES")

        if (
            self.props.tooth_taper != 0
            and self.props.height != 0
            and self.props.tooth_mod_enabled
        ):

            def _radial_lenth(v):
                return (v.co * Vector((1, 1, 0))).length

            tooth_tip_verts = [
                vert for vert in bm.verts if vert[self.tooth_tips_vert_grp] == 1
            ]
            root_ref_vert = next(
                vert for vert in ref_tooth_inside_verts if vert.is_valid
            )

            if is_internal:
                tip_ref_vert = sorted(tooth_tip_verts, key=_radial_lenth)[0]
            else:
                tip_ref_vert = sorted(tooth_tip_verts, key=_radial_lenth)[-1]

            taper_dist = min(self.props.tooth_taper, (self.props.height / 2) - 0.0001)
            base_xform = Vector((0, 0, -taper_dist))
            height = self.props.height
            scaler = Vector((1, 1, 0))
            tip_radius = (tip_ref_vert.co * scaler).length
            root_radius = (root_ref_vert.co * scaler).length
            offset = self.props.tooth_taper_offset

            if self.props.helix_angle == 0:
                for v in bm.verts:
                    radius = _radial_lenth(v)

                    if is_internal:
                        radial_scaler = map_range(
                            radius, (tip_radius, root_radius - offset), (1, 0)
                        )
                    else:
                        radial_scaler = map_range(
                            radius, (root_radius + offset, tip_radius), (0, 1)
                        )

                    height_scaler = (v.co.z / height * 2) - 1
                    v.co += base_xform * radial_scaler * height_scaler
            else:
                teeth_face_sets = self.get_teeth_face_sets(bm)
                teeth_face_sets.pop(0)
                teeth_face_sets.reverse()
                last = teeth_face_sets.pop()
                teeth_face_sets.insert(0, last)

                if is_internal:
                    cut_radius = root_radius - offset
                else:
                    cut_radius = root_radius + offset

                locs, norms, tangents = circle_points(
                    self.props.teeth, cut_radius, radians(90)
                )

                # bpy.context.scene.cursor.location = Vector((tip_radius, 0, 0))
                taper_length = abs(cut_radius - tip_radius)
                angle = self.props.tooth_taper / taper_length

                top_rot_angle = ((2 * pi) * height * tan(helix_angle)) / (
                    module * n_teeth * pi
                )
                top_rotation = Euler((0, 0, top_rot_angle))

                for loc, norm, tangent, face_set in zip(
                    locs, norms, tangents, teeth_face_sets
                ):
                    # Cut bottom
                    rotation = Quaternion(-tangent, (pi / 2) - angle)

                    if is_internal:
                        rotation = Quaternion(tangent, (pi / 2) - angle)
                        norm *= -1

                    norm.rotate(rotation)
                    geom = face_set + unique_face_edges(face_set)
                    result = bisect_geometry(bm, geom, loc=loc, norm=norm)
                    above_verts = result["above"]
                    below_verts = result["below"]
                    cut_geo = result["cut"]

                    bmesh.ops.delete(bm, geom=above_verts, context="VERTS")
                    cut_edges = [i for i in cut_geo if isinstance(i, BMEdge)]
                    bmesh.ops.edgenet_fill(bm, edges=cut_edges)

                    loc.z = height
                    if not self.props.herringbone:
                        loc.rotate(top_rotation)
                        norm.rotate(top_rotation)

                    norm.z *= -1
                    geom_faces = unique_vert_faces(below_verts)
                    geom_edges = unique_face_edges(geom_faces)
                    geom = geom_faces + geom_edges
                    result = bisect_geometry(bm, geom, loc=loc, norm=norm)
                    above_verts = result["above"]
                    cut_geo = result["cut"]
                    bmesh.ops.delete(bm, geom=above_verts, context="VERTS")
                    cut_edges = [i for i in cut_geo if isinstance(i, BMEdge)]
                    bmesh.ops.edgenet_fill(bm, edges=cut_edges)

        if self.props.worm_cut != 0 and self.props.tooth_mod_enabled:
            self.worm_wheel_cut(bm)

        # Crowning
        if self.props.tooth_mod_enabled:
            if self.props.flank_modification == "CROWNING":
                self.add_flank_crowing(bm)

            # End Relief
            if self.props.flank_modification == "END_RELIEF":
                self.add_flank_end_relief(bm)

        self.shade_smooth(bm)
        bm.to_mesh(mesh)
        bm.free()
        return mesh

    def worm_wheel_cut(self, bm: BMesh):
        def _radial_lenth(v):
            return (v.co * Vector((1, 1, 0))).length

        # Worm cut scale
        pitch_radius = self.props.module * self.props.teeth / 2
        root_radius = pitch_radius - (self.props.module * 1.24)  # Slightly offset from true root radius
        tip_radius = pitch_radius + self.props.module

        cut_distance = self.props.worm_cut
        base_xform = Vector((0, 0, cut_distance))
        scaler = Vector((1, 1, 0))

        amnt = self.props.worm_cut
        bias = self.props.worm_cut_scale
        bezier_res = self.props.z_resolution * 4

        ref_bezier = interpolate_bezier(
            Vector((0, 0, 0)),
            Vector((amnt, 0, 0)),
            Vector((amnt, 0, self.props.height)),
            Vector((0, 0, self.props.height)),
            bezier_res,
        )

        height_scaler = self.props.worm_cut_scale - 1
        interp_x = np.linspace(
            0 - (self.props.height * height_scaler),
            self.props.height + height_scaler,
            bezier_res,
        )
        interp_y = [v.x for v in ref_bezier]

        for vert in bm.verts:
            if _radial_lenth(vert) < root_radius:
                continue

            normal = (vert.co * Vector((1, 1, 0))).normalized() * -1
            interp_val = vert.co.z
            mag_scale = np.interp(interp_val, interp_x, interp_y)
            vert.co += normal * mag_scale

    def get_teeth_face_sets(self, bm: BMesh) -> List[List[BMFace]]:
        """Get lists of faces for each tooth. Index zero is non-tooth geo"""
        teeth_sets: List[List[BMFace]] = []
        for _ in range(self.props.teeth + 1):
            teeth_sets.append([])
        for face in bm.faces:
            tooth_id = face[self.tooth_id_face_grp]
            teeth_sets[tooth_id].append(face)

        return teeth_sets


class InvoluteGearRack(GearBuilder):
    type = "RACK"

    def __init__(self, props: PrecisionGearsProps):
        self.props = props
        self.rack = InvoluteRack(
            m=props.module,
            # z=props.teeth,
            z=1,
            pressure_angle=props.pressure_angle,
            thickness=props.width,
            beta=props.helix_angle,
            head=props.head,
            clearance=props.clearance,
            properties_from_tool=False,
            add_endings=True,
            simplified=False,
        )

    def create_gear(self) -> Mesh:
        bm = bmesh.new()
        mesh = bpy.data.meshes.new(self.type)
        rack_base_vert_grp = bm.verts.layers.int.new("rack_base")
        self.initialize_bmesh_layers(bm)

        pts = self.rack.points()
        pts = pts[1::]

        # Check for nan in points
        for co in pts:
            for value in co:
                if isnan(value):
                    print(co, "Impossible value detected")
                    return None

        pts = [Vector(pt).to_3d() for pt in pts]
        first_tooth_verts = [bm.verts.new(pt) for pt in pts]

        # Tooth Tips
        tip_verts = first_tooth_verts[2:4]
        for vert in tip_verts:
            vert[self.tooth_tips_vert_grp] = 1

        # Tooth root
        root_verts = [first_tooth_verts[1], first_tooth_verts[4]]
        for vert in root_verts:
            vert[self.tooth_root_vert_grp] = 1

        # Group flank A
        for vert in first_tooth_verts[1:3]:
            vert[self.flank_a_vert_grp] = 1

        # Group flank B
        for vert in first_tooth_verts[3:5]:
            vert[self.flank_b_vert_grp] = 1

        # Group base
        base_points = first_tooth_verts[-2:]
        for vert in base_points:
            vert[rack_base_vert_grp] = 1

        tooth_face = bm.faces.new([first_tooth_verts[i] for i in (1, 2, 3, 4)])
        tooth_face[self.tooth_id_face_grp] = 1
        base_face = bm.faces.new([first_tooth_verts[i] for i in (0, 1, 4, 5, 6, 7)])
        base_face[self.tooth_id_face_grp] = 0

        if self.props.tooth_mod_enabled:
            if self.props.tip_relief != 0:
                if self.props.tooth_mod_enabled:
                    amount = self.props.tip_relief
                    segments = self.props.tip_relief_resolution
                    bmesh.ops.bevel(
                        bm,
                        geom=tip_verts,
                        offset=amount,
                        segments=segments,
                        profile=0.5,
                    )

            if self.props.root_relief != 0:
                if self.props.tooth_mod_enabled:
                    # TODO: Doesn't play nice with flank modifications
                    amount = self.props.root_relief
                    segments = self.props.root_relief_resolution
                    bmesh.ops.bevel(
                        bm,
                        geom=root_verts,
                        offset=amount,
                        segments=segments,
                        profile=0.5,
                    )

        first_tooth_verts = [vert for vert in bm.verts[:]]
        first_tooth_faces = [face for face in bm.faces[:]]

        # Duplicate teeth
        section_length = (base_points[0].co - base_points[1].co).length
        if self.props.teeth > 1:
            bmesh.ops.spin(
                bm,
                geom=bm.faces,
                dvec=Vector((0, section_length, 0)),
                use_duplicate=True,
                use_merge=True,
                steps=self.props.teeth - 1,
            )
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)

        if self.props.height == 0:
            self.shade_smooth(bm)
            bm.to_mesh(mesh)
            return mesh
        elif self.props.helix_angle == 0:
            # steps = self.props.z_resolution
            steps = self.props.z_resolution
            step_xform = Vector((0, 0, self.props.height / steps))
            bmesh.ops.spin(
                bm, geom=bm.faces, dvec=step_xform, use_duplicate=False, steps=steps
            )
        elif self.props.herringbone:
            y_xform = np.tan(self.props.helix_angle) * self.props.height
            xform = Vector((0, y_xform, self.props.height))
            # steps = self.props.z_resolution
            steps = self.props.z_resolution
            step_xform = xform * (1 / steps) * 0.5
            result = bmesh.ops.spin(
                bm,
                geom=bm.faces,
                dvec=step_xform,
                use_duplicate=False,
                steps=steps,
            )
            mid_faces = [
                elem for elem in result["geom_last"] if isinstance(elem, BMFace)
            ]

            result = bmesh.ops.spin(
                bm,
                geom=mid_faces,
                dvec=step_xform * Vector((1, -1, 1)),
                use_duplicate=False,
                steps=steps,
            )
            bmesh.ops.delete(bm, geom=mid_faces, context="FACES_KEEP_BOUNDARY")
        else:
            y_xform = np.tan(self.props.helix_angle) * self.props.height
            xform = Vector((0, y_xform, self.props.height))
            steps = self.props.z_resolution
            step_xform = xform * (1 / steps)
            bmesh.ops.spin(
                bm,
                geom=bm.faces,
                dvec=step_xform,
                use_duplicate=False,
                steps=steps,
            )

        if self.props.tooth_taper != 0 and self.props.height != 0:
            if self.props.tooth_mod_enabled:
                ref_root = next(
                    vert for vert in bm.verts if vert[self.tooth_root_vert_grp] == 1
                )
                taper_root = ref_root.co * Vector((1, 1, 0))
                taper_root.x += self.props.tooth_taper_offset
                bisect_no = Vector((1, 0, 0))
                geom = bm.faces[:] + bm.edges[:]
                result = bisect_geometry(bm, geom, loc=taper_root, norm=bisect_no)

                taper_verts = [vert for vert in bm.verts if vert.co.x > taper_root.x]

                taper_dist = min(
                    self.props.tooth_taper, (self.props.height / 2) - 0.0001
                )

                base_xform = Vector((0, 0, -taper_dist))
                height = self.props.height
                first_tooth_verts = [
                    vert for vert in first_tooth_verts if vert.is_valid
                ]
                max_x = sorted(first_tooth_verts, key=lambda v: v.co.x)[-1].co.x

                if self.props.helix_angle != 0:
                    rot_offset = Euler((-self.props.helix_angle, 0, 0))
                    base_xform.rotate(rot_offset)
                    base_xform.normalize()
                    length = self.props.tooth_taper / cos(self.props.helix_angle)
                    base_xform *= length

                y_flipped_xform = base_xform.copy()
                y_flipped_xform.y *= -1

                for vert in taper_verts:
                    x = vert.co.x
                    dist_scaler = map_range(x, (taper_root.x, max_x), (0, 1))
                    height_scaler = (vert.co.z / height * 2) - 1
                    if self.props.herringbone and vert.co.z > height / 2:
                        vert.co += y_flipped_xform * dist_scaler * height_scaler
                    else:
                        vert.co += base_xform * dist_scaler * height_scaler

        # Crowning
        if self.props.tooth_mod_enabled:
            if self.props.flank_modification == "CROWNING":
                self.add_flank_crowing(bm)

            # End Relief
            if self.props.flank_modification == "END_RELIEF":
                self.add_flank_end_relief(bm)

        self.shade_smooth(bm)
        bm.to_mesh(mesh)
        return mesh

    def add_flank_end_relief(self, bm):
        if self.props.tooth_taper == 0:
            super().add_flank_end_relief(bm)
            return None

        # TODO: Root relief causes misalignment of cut origin
        # TODO: Taper angle slightly off. Probably related to cut origin.

        if self.props.flank_mod_param_a == 0 or self.props.flank_mod_param_a == 0:
            return None

        relief_amount = self.props.flank_mod_param_a
        relief_dist = self.props.flank_mod_param_b

        relief_dist = min(
            self.props.flank_mod_param_b, (self.props.height / 2) - 0.0001
        )

        cut_top_z = self.props.height - relief_dist
        cut_bot_z = relief_dist
        teeth_faces = [face for face in bm.faces if face[self.tooth_id_face_grp] == 1]
        teeth_verts = unique_face_verts(teeth_faces)

        x_sorted_verts = sorted(teeth_verts, key=lambda v: v.co.x)
        tooth_min_x = x_sorted_verts[0].co.x
        tooth_max_x = x_sorted_verts[-1].co.x

        taper_dist = min(self.props.tooth_taper, (self.props.height / 2) - 0.0001)

        taper_offset = self.props.tooth_taper_offset
        taper_origin_x = tooth_min_x + taper_offset - 0.00001
        tapered_verts = [vert for vert in teeth_verts if vert.co.x >= taper_origin_x]
        tapered_faces = unique_vert_faces(tapered_verts)
        root_faces = list(set(bm.faces).difference(tapered_faces))

        ref_root_face = next(
            face for face in bm.faces if face[self.tooth_id_face_grp] == 0
        )

        taper_angle = taper_dist / abs((tooth_max_x - taper_origin_x))

        cut_top_loc = Vector((taper_origin_x, 0, cut_top_z))
        cut_bot_loc = Vector((taper_origin_x, 0, cut_bot_z))
        cut_locs = (cut_top_loc, cut_bot_loc)

        cut_top_norm = Vector((0, 0, 1))
        cut_top_norm.rotate(Euler((0, taper_angle, 0)))
        cut_bot_norm = Vector((0, 0, -1))
        cut_bot_norm.rotate(Euler((0, -taper_angle, 0)))
        cut_norms = (cut_top_norm, cut_bot_norm)

        tapered_geo = tapered_faces + unique_face_edges(tapered_faces)
        relief_verts = set()

        # Tooth Cuts
        for loc, norm in zip(cut_locs, cut_norms):
            result = bisect_geometry(bm, geom=tapered_geo, loc=loc, norm=norm)
            relief_verts = relief_verts.union(result["above"])
            result_verts = dict_by_type(chain(*result.values()))[BMVert]
            tapered_geo = unique_vert_faces(result_verts) + unique_vert_edges(
                result_verts
            )

        relief_faces = set(dict_by_type(tapered_geo)[BMFace])
        root_faces = set(bm.faces[:]).difference(relief_faces)

        root_geo = list(root_faces) + unique_face_edges(root_faces)
        cut_locs = reversed(cut_locs)

        for loc in cut_locs:
            result = bisect_geometry(
                bm, geom=root_geo, loc=loc, norm=Vector((0, 0, -1))
            )
            result_verts = dict_by_type(chain(*result.values()))[BMVert]
            root_geo = unique_vert_faces(result_verts)
            root_geo += unique_face_edges(root_geo)

        relief_vert_normals = calc_vertex_normals(relief_verts)

        for vert, normal in zip(relief_verts, relief_vert_normals):
            signed_angle = normal.xy.angle_signed(Vector((1, 0)))
            if signed_angle > 0 + radians(1):
                vert.co.y -= relief_amount
            elif signed_angle < 0 - radians(1):
                vert.co.y += relief_amount
            else:
                continue

        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.001)


class BevelGear(GearBuilder):
    type = "BEVEL"

    def __init__(self, props: PrecisionGearsProps):
        self.props = props
        self.involute_tooth = InvoluteTooth(
            m=props.module,
            z=props.teeth,
            pressure_angle=props.pressure_angle,
            clearance=props.clearance,
            shift=0,
            beta=props.helix_angle,
            undercut=False,
            backlash=props.backlash,
            head=props.head,
            properties_from_tool=False,
        )

    @cached_property
    def pitch_circle_radius(self) -> float:
        pitch_circle_radius = (self.props.teeth * self.props.module) / 2
        return pitch_circle_radius

    @cached_property
    def pitch_apex(self) -> Vector:
        # dedendum = self.props.module * 1.25
        z = self.pitch_circle_radius /  tan(self.props.pitch_angle)
        return Vector((0, 0, z))

    def create_gear(self) -> Mesh:
        bm = bmesh.new()
        self.initialize_bmesh_layers(bm)
        mesh = bpy.data.meshes.new(self.type)

        try:
            pts = self.involute_tooth.points(num=self.props.complexity)
            n_teeth = self.props.teeth

            # Check for nan in points
            for co in itertools.chain.from_iterable(pts):
                for value in co:
                    if isnan(value):
                        print(co, "Impossible value detected")
                        return None

            # Ensure recognized point counts
            if len(pts) == 5:
                side_a_pts = list(chain(*pts[0:2]))
                side_b_pts = list(chain(*pts[3:]))
                side_b_pts.reverse()
            elif len(pts) == 3:
                side_a_pts = list(chain(pts[0]))
                side_b_pts = list(chain(pts[2]))
                side_b_pts.reverse()
            else:
                print(f"Unrecognized involute tooth point set count {len(pts)}")
                return None

            # Mesh first tooth
            ref_tooth_verts: List[BMVert] = []
            ref_tooth_faces: List[BMFace] = []
            for side_a, side_b in zip(side_a_pts, side_b_pts):
                new_vert_a = bm.verts.new(Vector(side_a).to_3d())
                new_vert_b = bm.verts.new(Vector(side_b).to_3d())

                if new_vert_a.co.length >= self.pitch_circle_radius:
                    new_vert_a[self.module_vert_grp] = 1
                    new_vert_b[self.module_vert_grp] = 1

                new_vert_a[self.flank_a_vert_grp] = 1
                new_vert_b[self.flank_b_vert_grp] = 1
                if ref_tooth_verts:
                    face = bm.faces.new([*ref_tooth_verts[-2:], new_vert_b, new_vert_a])
                    face[self.tooth_id_face_grp] = 1
                    face.normal_flip()
                    ref_tooth_faces.append(face)

                ref_tooth_verts.append(new_vert_a)
                ref_tooth_verts.append(new_vert_b)

            ref_root_avg = (ref_tooth_verts[0].co + ref_tooth_verts[1].co) * 0.5
            # self.pitch_circle_radius
            cent = Vector((self.pitch_circle_radius, 0, 0))

            # TODO: Check if this is correct profile and if so replace with cutter at pitch angle
            pitch_euler = Euler((0, -self.props.pitch_angle, 0))
            pitch_m = pitch_euler.to_matrix()
            bmesh.ops.rotate(
                bm,
                cent=cent,
                matrix=pitch_m,
                verts=bm.verts,
            )

            for vert in ref_tooth_verts[:2]:
                vert[self.tooth_root_vert_grp] = 1

            # Identify tooth features
            ref_tooth_tip_verts = ref_tooth_verts[-2:]
            for vert in ref_tooth_tip_verts:
                vert[self.tooth_tips_vert_grp] = 1

            # Identify ref inside verts
            ref_tooth_inside_verts = ref_tooth_verts[:2]
            for vert in ref_tooth_inside_verts:
                vert[self.tooth_inside_vert_grp] = 1

            # Assign index to ref tooth
            for vert in ref_tooth_verts:
                vert[self.tooth_index_vert_grp] = 1
                vert[self.bottom_verts_grp] = 1

            # Align teeth faces to tangent of pitch angle
            # max_x = sorted([v.co.x for v in bm.verts])[-1]
            # for vert in bm.verts:
            #     xz = vert.co * Vector((1, 0, 1))
            #     apex_xz = self.pitch_apex * Vector((1, 0, 1))
            #     n = (xz - apex_xz).normalized()
            #     distance = max_x - xz.x
            #     vert.co += n * distance
            #     # n = self.pitch_apex.xz

            # Extrude first tooth
            extrusion_target = self.pitch_apex
            geom = bm.faces[:] + bm.edges[:]
            extrusion = bmesh.ops.extrude_face_region(bm, geom=geom)["geom"]
            extrusion_verts = [elem for elem in extrusion if isinstance(elem, BMVert)]
            
            # Group extruded
            for vert in extrusion_verts:
                vert[self.top_verts_grp] = 1
                vert[self.bottom_verts_grp] = 0

            for vert in extrusion_verts:
                direction = extrusion_target - vert.co
                max_extrusion = direction.length
                direction.normalize()
                offset = direction * self.props.height
                if offset.length > max_extrusion:
                    offset = offset.normalized() * max_extrusion
                vert.co += offset

            # Delete bottom face
            root_verts = set(verts_by_layer(bm.verts, self.tooth_root_vert_grp))
            for face in bm.faces:
                face_verts = set(face.verts)
                if face_verts == root_verts:
                    bm.faces.remove(face)
                    break

            # raise ValueError
            # Slice teeth
            side_a_verts = verts_by_layer(bm.verts, self.flank_a_vert_grp)
            side_b_verts = verts_by_layer(bm.verts, self.flank_b_vert_grp)
            side_a_edges = shared_edges(side_a_verts)
            side_b_edges = shared_edges(side_b_verts)
            side_edges = set(side_a_edges + side_b_edges)

            top_verts = verts_by_layer(bm.verts, self.top_verts_grp)
            bottom_verts = verts_by_layer(bm.verts, self.bottom_verts_grp)
            top_edges = shared_edges(top_verts)
            bottom_edges = shared_edges(bottom_verts)
            cap_edges = set(top_edges + bottom_edges)

            side_edges = side_edges.difference(cap_edges)

            cuts = self.props.z_resolution
            edges = list(side_edges)
            result = bmesh.ops.subdivide_edges(bm, edges=edges, cuts=cuts, smooth=0)
            type_sorted_result = dict_by_type(result["geom"])
            for vert in type_sorted_result[BMVert]:
                vert[self.top_verts_grp] = 0
                vert[self.bottom_verts_grp] = 0

            # Apply helix angle
            if self.props.helix_angle != 0:
                for vert in bm.verts:
                    distance = (self.pitch_apex - vert.co).length
                    angle = self.props.helix_angle * distance
                    axis = Vector((0, 0, 1))
                    m = Matrix.Rotation(angle, 4, axis)
                    bmesh.ops.rotate(bm, cent=self.pitch_apex, verts=[vert], matrix=m)

            # Create teeth from reference tooth
            axis = Vector((0, 0, 1))
            step_angle = 2 * pi / n_teeth

            ref_tooth_faces = bm.faces[:]
            first_tooth_verts = verts_from_faces(ref_tooth_faces)
            prev_tooth_verts = verts_from_faces(ref_tooth_faces)

            def _connect_teeth(from_tooth_verts, to_tooth_verts):
                from_verts = list(
                    verts_by_layer(
                        from_tooth_verts, (self.tooth_root_vert_grp, self.flank_b_vert_grp)
                    )
                )
                to_verts = list(
                    verts_by_layer(
                        to_tooth_verts, (self.tooth_root_vert_grp, self.flank_a_vert_grp)
                    )
                )

                from_edges = shared_edges(from_verts)
                to_edges = shared_edges(to_verts)
                bmesh.ops.bridge_loops(bm, edges=from_edges + to_edges)
            
            # Create teeth
            for tooth_id in range(1, n_teeth):
                angle = step_angle * tooth_id
                result = bmesh.ops.spin(
                    bm,
                    geom=ref_tooth_faces,
                    angle=angle,
                    axis=axis,
                    steps=1,
                    use_duplicate=True,
                )
                type_sorted_result = dict_by_type(result["geom_last"])

                # Assign per tooth verts
                for face in type_sorted_result[BMFace]:
                    face[self.tooth_id_face_grp] = tooth_id + 1
                for vert in type_sorted_result[BMVert]:
                    vert[self.tooth_index_vert_grp] = tooth_id + 1

                # Connect teeth
                if tooth_id != n_teeth - 1:
                    _connect_teeth(prev_tooth_verts, type_sorted_result[BMVert])
                else:
                    _connect_teeth(type_sorted_result[BMVert], first_tooth_verts)
                    _connect_teeth(prev_tooth_verts, type_sorted_result[BMVert])

                prev_tooth_verts = type_sorted_result[BMVert]

            # Create center faces
            for grp in (self.top_verts_grp, self.bottom_verts_grp):
                verts = verts_by_layer(bm.verts, (self.tooth_inside_vert_grp, grp))
                sorted_verts = polar_sort_verts(list(verts))
                bm.faces.new(sorted_verts)
        except Exception as e:
            print(e)
        finally:
            # Apply to mesh and return
            bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0001)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            bm.to_mesh(mesh)
            bm.free

        return mesh


class WormGear(GearBuilder):
    type = "WORM"

    def __init__(self, props: PrecisionGearsProps):
        self.props = props

    def create_gear(self):
        module = self.props.module
        diameter = self.props.diameter
        teeth = self.props.teeth
        height = self.props.height
        mesh = bpy.data.meshes.new(self.type)
        pitch = module * pi

        bm = bmesh.new()

        try:
            clearance = self.props.clearance
            head = self.props.head
            pressure_angle = self.props.pressure_angle

            r_1 = (diameter - (2 + 2 * clearance) * module) / 2
            r_2 = (diameter + (2 + 2 * head) * module) / 2
            z_a = (2 + head + clearance) * module * np.tan(pressure_angle)
            z_b = (module * np.pi - 4 * module * np.tan(pressure_angle)) / 2
            z_0 = clearance * module * np.tan(pressure_angle)
            z_1 = z_b - z_0
            z_2 = z_1 + z_a
            z_3 = z_2 + z_b - 2 * head * module * np.tan(pressure_angle)
            z_4 = z_3 + z_a

            x1 = r_1
            x2 = r_2

            locs = []

            # Add a extra extra point for correct shape extrapolation
            init_z = -(pitch * (teeth + 1))
            locs.append(Vector((x1, 0, init_z + z_4 - pitch)))
            n_loops = ceil(height / (pitch * teeth)) * (teeth * 2)  # This count's buffer amount is jank
            n_loops += 3
            for tooth in range(n_loops):
                z_offset = pitch * tooth
                locs.extend((
                    Vector((x1, 0, init_z + z_1 + z_offset)),
                    Vector((x2, 0, init_z + z_2 + z_offset)),
                    Vector((x2, 0, init_z + z_3 + z_offset)),
                    Vector((x1, 0, init_z + z_4 + z_offset)),
                ))

            # Create profile edges
            verts = [bm.verts.new(l) for l in locs]
            for v1, v2 in zip(verts, verts[1:]):
                bm.edges.new((v1, v2))

            axis = Vector((0, 0, 1))
            steps = self.props.z_resolution
            angle = radians(360)
            if self.props.reverse_pitch:
                angle = -angle
            height_per_step = (pitch * teeth) / steps
            height_per_step = Vector((0, 0, height_per_step))

            bmesh.ops.spin(
                bm,
                geom=bm.edges[:],
                axis=axis,
                dvec=height_per_step,
                angle=angle,
                steps=steps,
                use_duplicate=False,
            )

            bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.001)

            # Remove geometry above height argument
            p, n = Vector((0, 0, height)), Vector((0, 0, 1))  # Cut location and normal
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            bisection = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=p, plane_no=n, clear_outer=True)
            cut_edges = dict_by_type(bisection["geom_cut"])[BMEdge]
            cap_faces = bmesh.ops.edgenet_fill(bm, edges=cut_edges)["faces"]
            bmesh.ops.poke(bm, faces=cap_faces)

            # Remove geometry below zero
            p, n = Vector((0, 0, 0)), Vector((0, 0, -1))  # Cut location and normal
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            bisection = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=p, plane_no=n, clear_outer=True)
            cut_edges = dict_by_type(bisection["geom_cut"])[BMEdge]
            cap_faces = bmesh.ops.edgenet_fill(bm, edges=cut_edges)["faces"]
            bmesh.ops.poke(bm, faces=cap_faces)

            for f in bm.faces:
                f.smooth = True
        except Exception as e:
            print(e)
        finally:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            bm.to_mesh(mesh)
            return mesh


class GT2Gear(GearBuilder):
    # TODO: Build this shit
    type = "GT2GEAR"
    profile_sections = (
        [(-0.0, 0.0), (0.165, 0.0), (0.415, 0.076), (0.532, 0.391)],
        [(0.532, 0.391), (0.596, 0.564), (0.583, 0.646), (0.618, 0.692)],
        [(0.618, 0.692), (0.644, 0.726), (0.688, 0.755), (0.742, 0.755)],
    )

    def __init__(self, props: PrecisionGearsProps):
        self.props = props

    @property
    def _profile_sections_as_vectors(self):
        for section in self.profile_sections:
            yield [Vector(v).to_3d() for v in section]

    def create_gear(self):
        mesh = bpy.data.meshes.new(self.type)
        bm = bmesh.new()
        pitch = 2.0
        circumference = self.props.teeth * pitch
        pitch_radius = circumference / tau
        radius = pitch_radius - 1.38

        try:
            # Create first tooth profile
            profile_res = max(self.props.complexity, 2)
            profile = deque()
            for section in self._profile_sections_as_vectors:
                points = interpolate_bezier(*section, profile_res)
                for p in points:
                    p.y += radius
                    profile.appendleft(p)
                    mirrored = p.copy() * Vector((-1, 1, 1))
                    profile.append(mirrored)
            
            # Build verts and edges from profile points 
            last_co = profile.popleft()
            last_v = bm.verts.new(last_co)
            while profile:
                next_co = profile.popleft()
                # Create new vert and edge only if distance would be more than 0.01
                co_dist = (last_co - next_co).length
                if co_dist > 0.01:
                    next_v = bm.verts.new(next_co)
                    bm.edges.new((last_v, next_v))
                    last_co = next_co
                    last_v = next_v
            n_profile_points = len(bm.verts)
            
            # Duplicate teeth
            angle = tau - (tau / self.props.teeth)
            bmesh.ops.spin(
                bm,
                geom=bm.edges,
                steps=self.props.teeth - 1,
                angle=angle,
                axis=(0.0, 0.0, 1.0),
                use_duplicate=True,
            )

            # Create edge between adjacent teeth
            bm.verts.ensure_lookup_table()
            for i in range(0, len(bm.verts), n_profile_points):
                v1 = bm.verts[i -1]
                v2 = bm.verts[i]
                bm.edges.new((v1, v2))

            # Extrude profile upwards by half height
            ret = bmesh.ops.extrude_edge_only(bm, edges=bm.edges)
            extruded = dict_by_type(ret["geom"])
            xform = Vector((0, 0, self.props.height / 2))
            bmesh.ops.translate(bm, verts=extruded[BMVert], vec=xform)

            # Create teeth inner faces
            if self.props.head > 0.0:
                ret = bmesh.ops.extrude_edge_only(bm, edges=extruded[BMEdge])
                extruded = dict_by_type(ret["geom"])
                for v in extruded[BMVert]:
                    xy = v.co.xy.normalized()
                    xy *= pitch_radius + self.props.clearance
                    v.co.xy = xy

                # Create top section
                ret = bmesh.ops.extrude_edge_only(bm, edges=extruded[BMEdge])
                extruded = dict_by_type(ret["geom"])
                xform = Vector((0, 0, self.props.head))
                bmesh.ops.translate(bm, verts=extruded[BMVert], vec=xform)
            #     bmesh.ops.edgeloop_fill(bm, edges=extruded[BMEdge])
            # else:

            bmesh.ops.edgeloop_fill(bm, edges=extruded[BMEdge])


            # Mirror geometry
            geom = bm.faces[:] + bm.verts[:]
            bmesh.ops.mirror(bm, geom=geom, axis="Z", merge_dist=0.1)

        except Exception as e:
            print(e)
        finally:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            bm.to_mesh(mesh)
            return mesh