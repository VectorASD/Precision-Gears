from dataclasses import dataclass, field
from typing import Union, Dict

import bpy
from bpy.types import Mesh, Object
from mathutils import Vector

from . import mesh_generation
from . import mesh_editing


@dataclass
class BoreGeo():
    size: float            # Size centerpiece. For square objects furthest face. For round, radius
    depth: float           # Depth of generated geometry
    param_a: float         # Type specific parameter
    param_b: float         # Type specific parameter
    resolution: int        # Resolution for applicable types
    subtype: bool = False  # Mode switch for type
    _mesh: Mesh = None
    _object: Object = None
    type: str = field(init=False)
    name: str = field(init=False)
    prop_names: Union[Dict[str, str], None] = None

    def create(self) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        if self._mesh is not None:
            bpy.data.meshes.remove(self._mesh)

        if self._object is not None:
            bpy.data.objects.remove(self._object)

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            self.create()
        return self._mesh

    @property
    def object(self) -> Object:
        if self._object is None:
            self._object = bpy.data.objects.new(name=self.type, object_data=self.mesh)
        return self._object


class Cross(BoreGeo):
    """
    A Cross bore. Variable extrusion lengths. Extrusion axis can be disabled
    No subtype

    Attribute Use
    size: Size of center square
    depth: Depth of generated geometry
    param_a: X extrusion size
    param_b: Y extrusion Size
    """
    type = "CROSS"
    name = "Cross"
    prop_names = {
        "bore_size": "Core Size",
        "bore_param_a": "X Size",
        "bore_param_b": "Y Size",
    }

    def create(self) -> None:
        core = mesh_generation.new_cross_mesh(
            self.size,
            x_length=self.param_a,
            y_length=self.param_b,
            depth=self.depth
        )
        self._mesh = core


class Round_A(BoreGeo):
    """
    A cylindrical with square subtractions

    Attribute Use
    size: Radius of cylinder
    depth: Depth of generated geometry
    param_a: Cutout offset
    param_b: Cutout scale
    subtype: Mirroring of cutouts
    """
    type = "ROUND_SUBBED"
    name = "Round Subtract"
    prop_names = {
        "bore_size": "Radius",
        "bore_param_a": "Cutter Position",
        "bore_param_b": "Cutter Scale",
        "bore_resolution": "Resolution",
        "bore_subtype": "Mirror",
    }

    def create(self) -> None:
        core = mesh_generation.new_cylinder_mesh(
            self.size, depth=self.depth, segments=self.resolution)
        # extrusion_scale_factor = self.size * 1
        extrusion_size = Vector((
            self.param_b,
            self.param_b,
            self.depth * 2
        ))
        extrusion = mesh_generation.new_cube_mesh(extrusion_size)
        extrusion_loc = Vector((self.param_a, 0, 0))
        core = mesh_editing.boolean_mesh(
            core, extrusion, operation="DIFFERENCE", location=extrusion_loc)

        if self.subtype:
            core = mesh_editing.boolean_mesh(
                core, extrusion, operation="DIFFERENCE", location=extrusion_loc * -1)

        self._mesh = core


class Round_B(BoreGeo):
    """
    A cylindrical with square extrusions

    Attribute Use
    size: Radius of cylinder
    depth: Depth of generated geometry
    param_a: Cutout offset
    param_b: Cutout size
    subtype: Mirroring of cutouts
    """
    type = "ROUND_UNION"
    name = "Round Add"
    prop_names = {
        "bore_size": "Radius",
        "bore_param_a": "Union Position",
        "bore_param_b": "Union Scale",
        "bore_resolution": "Resolution",
        "bore_subtype": "Mirror",
    }

    def create(self) -> None:
        core = mesh_generation.new_cylinder_mesh(
            self.size, depth=self.depth, segments=self.resolution)
        extrusion_scale_factor = self.size * 1
        extrusion_size = Vector((
            self.param_b,
            self.param_b,
            self.depth * 2
        ))
        extrusion = mesh_generation.new_cube_mesh(extrusion_size)
        extrusion_loc = Vector((self.param_a, 0, 0))

        core = mesh_editing.boolean_mesh(
            core, extrusion, operation="UNION", location=extrusion_loc)

        if self.subtype:
            core = mesh_editing.boolean_mesh(
                core, extrusion, operation="UNION", location=extrusion_loc * -1
            )

        self._mesh = core


class Polygonal(BoreGeo):
    """
    Polygonal bore, such as a hex

    Attribute Use
    size: Radius of bore
    depth: Depth of generated geometry
    param_a: Polygon side
    """
    type = "POLYGON"
    name = "Polygon"
    prop_names = {
        "bore_size": "Radius",
        "bore_resolution": "Resolution",
    }


    def create(self) -> None:
        self.param_a = max(self.param_a, 3)

        core = mesh_generation.new_cylinder_mesh(
            self.size, segments=self.resolution, depth=self.depth)
        self._mesh = core


def _compose_enum():
    entries = []
    entries.append(("NONE", "None", "None", 0))
    for index, cls in enumerate(BoreGeo.__subclasses__()):
        entries.append((cls.type, cls.name, cls.name, index + 1))
    return entries


def get(of_type: str) -> Union[BoreGeo, None]:
    bores = list(BoreGeo.__subclasses__())
    return next((bore for bore in bores if bore.type == of_type), None)


bores = list(BoreGeo.__subclasses__())
enum = _compose_enum()
