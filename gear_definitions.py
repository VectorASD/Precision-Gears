"""
Interface between props and gears
"""
from __future__ import annotations
from math import pi, radians
import math
from typing import Any, Dict, NamedTuple, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass

import bpy
import bmesh
from mathutils import Vector, bvhtree
from bpy.types import Context, Mesh, Object, UILayout

from . import (
    bores,
    flank_mods,
    gear_builders,
    math_funcs,
    mesh_generation,
    operators,
    presets,
    consts,
    rigging,
    handle_autosmooth,
)
from .bmesh_helpers import angle_crease_mesh
from .consts import (
    GEAR_PROPS_ALIAS,
    GEAR_ATTRIBS_ALIAS,
    SET_SCREW_ARRANGEMENTS,
    PRESETS_DIR,
    BPY_4_1_COMPAT_REQUIRED,
)
from .gear_builders import GearBuilder
from .math_funcs import get_bounding_box, minify_bbox, mean_vectors
from .mesh_editing import boolean_mesh, floor_mesh, translate_mesh

if TYPE_CHECKING:
    from .props import PrecisionGearsProps


# def update_rig_attributes(self: PrecisionGearsProps, context: Context):
#     props = self
#     gear: Object = self.id_data
#     builder = get_gear_definition(props.gear_type)(props)
#     rig_attributes = builder.calculate_attributes()
#     gear[GEAR_ATTRIBS_ALIAS] = rig_attributes

# def update_shade_smooth(target: Object, angle: float):
#     print("updating_smooth_shading", target, angle)
#     if consts.BPY_4_1_COMPAT_REQUIRED:
#         override = bpy.context.copy()
#         override["object"] = target
#         override["active_object"] = target
#         override["selected_objects"] = [target,]
#         override["selected_editable_objects"] = [target,]
#         with bpy.context.temp_override(**override):
#             bpy.ops.object.shade_smooth_by_angle("INVOKE_DEFAULT", angle=angle, keep_sharp_edges=False)
#     else:
#         target.data.use_auto_smooth = True
#         target.data.auto_smooth_angle = angle

# def update_shade_smooth_callback(self, context: Context):
#     update_shade_smooth(context.active_object, self.smooth_angle)

def update_gear(self: PrecisionGearsProps, context: Context):
    props = self
    gear: Object = self.id_data

    # Editing flag is a sign that other callbacks are editing the property group
    if props.editing:  # Early return to prevent infinite recursion
        return None

    # Setting editing flag to allow value updating without infinite recursion
    props.editing = True

    # Ensure valid prop ranges
    try:
        # Build Gear
        builder = get_gear_definition(props.gear_type)(props)

        # Ensure valid prop ranges
        for prop_key in builder.get_prop_map().keys():
            min_val = builder.get_prop_map()[prop_key].min_val
            max_val = builder.get_prop_map()[prop_key].max_val
            current_val = getattr(props, prop_key)

            if min_val is not None:
                if min_val > current_val:
                    setattr(props, prop_key, min_val)
            if max_val is not None:
                if max_val < current_val:
                    setattr(props, prop_key, max_val)

        # Create gear geometry
        builder.create_gear()

        if props.gear_type != "BEVEL":
            builder.center_gear()

        gear_bbox = minify_bbox(get_bounding_box(builder.mesh))

        if builder.error_encountered:
            raise ValueError("Invalid Gear Generation Arguments")

        if (
            props.extrusion_size != 0
            and "extrusion_size" in builder.get_prop_map().keys()
        ):
            if props.extrusion_enabled:
                builder.add_extrusion(gear_bbox)

        if props.set_screw != 0 and "set_screw" in builder.get_prop_map().keys():
            if props.set_screw_enabled:
                builder.add_set_screw()

        if props.bore != "NONE" and "bore" in builder.get_prop_map().keys():
            if props.bore_enabled:
                builder.add_bore()

        if "do_crease" in builder.get_prop_map().keys():
            if props.do_crease:
                builder.crease_angles()

        if "add_weld" in builder.get_prop_map().keys():
            weld_modifier = gear.modifiers.get("Weld", None)
            if not props.add_weld and weld_modifier is not None:
                gear.modifiers.remove(weld_modifier)
            elif props.add_weld and weld_modifier is None:
                weld_modifier = gear.modifiers.new(name="Weld", type="WELD")
                weld_modifier.merge_threshold = props.weld_dist
            elif props.add_weld and weld_modifier is not None:
                weld_modifier.merge_threshold = props.weld_dist

        try:
            previous_mesh = gear.data
            gear.data = builder.mesh

            gear[GEAR_ATTRIBS_ALIAS] = builder.calculate_attributes()

            for f in gear.data.polygons:
                f.use_smooth = True

            if props.scaler != 1:
                for v in gear.data.vertices:
                    v.co *= props.scaler

            # Transfer materials
            for material in previous_mesh.materials:
                if material is not None:
                    gear.data.materials.append(material)

            bpy.data.meshes.remove(previous_mesh)

            handle_autosmooth.ensure_autosmooth_on_object(gear)

        except Exception as e:
            print(e)
            pass

    except Exception as e:
        print(e)
    finally:
        context.view_layer.update()
        context.view_layer.objects.active = gear
        props.uv_seam_angle = props.uv_seam_angle  # Trigger uv update

        # update_shade_smooth(gear, self.smooth_angle)

        props.editing = False  # Reset editing flag
        if props.rigging_enabled:
            pass
            # rigging.update_rig_node_props(self)
            props.refresh_rig = True


class PropertyMapping(NamedTuple):
    name: str  # Name of arg to use prop as on calculate function
    default: Any = None
    min_val: Union[None, Any] = None
    max_val: Union[None, Any] = None
    newline: bool = True


class GearCreator(ABC):
    gear_type: str = ""
    props: PrecisionGearsProps
    mesh: Union[None, Mesh] = None
    builder: GearBuilder
    centering_scaler = Vector((1, 1, 1))  # Scale centering vector by this limit effect
    error_encountered: bool = True
    gear_props: Union[dict, None] = None
    bore_props: Union[dict, None] = None
    set_screw_props: Union[dict, None] = None
    teeth_modification_props: Union[dict, None] = None
    extrusion_props: Union[dict, None] = None
    extra_props: Union[dict, None] = None
    toggle_props: Union[dict, None] = None

    _preset_scope_props = (
        "use_preset_base",
        "use_preset_tooth_mod",
        "use_preset_bore",
        "use_preset_extrusion",
        "use_preset_set_screw",
        "use_preset_general",
    )

    @classmethod
    def get_prop_map(cls) -> Dict[str, PropertyMapping]:
        """
        List of properties from props.PrecisionGearsProps to use and their constraints
        """
        prop_map = {}
        prop_map.update(cls.gear_props)
        if cls.bore_props is not None:
            prop_map.update(cls.bore_props)
        if cls.set_screw_props is not None:
            prop_map.update(cls.set_screw_props)
        if cls.extrusion_props is not None:
            prop_map.update(cls.extrusion_props)
        if cls.extra_props is not None:
            prop_map.update(cls.extra_props)
        if cls.teeth_modification_props is not None:
            prop_map.update(cls.teeth_modification_props)
        if cls.toggle_props is not None:
            prop_map.update(cls.toggle_props)
        return prop_map

    @classmethod
    def _draw_presets(cls, layout: UILayout, prop_grp):
        row = layout.row()
        box = row.box()
        col = box.column(align=True)

        # Presets Row
        row = col.row(align=True)

        # Thumb Presets Row
        row = col.row(align=True)
        col = box.column(align=True)
        row.label(text="Presets")
        apply_filter = getattr(prop_grp, "apply_preset_filter")

        if apply_filter:
            row.prop(prop_grp, "apply_preset_filter", icon="VIEWZOOM", text="")
        else:
            row.operator("object.apply_gear_preset_filter", icon="VIEWZOOM", text="")

        save_preset = row.operator(
            "object.save_gear_preset", icon="PRESET_NEW", text=""
        )

        # Oper User Preset Folder
        open_folder = row.operator(
            "wm.gears_open_sys_folder", icon="FILE_FOLDER", text=""
        )
        folder = consts.PRESETS_DIR
        open_folder.folder = str(folder)

        # Open user website
        open_store = row.operator("wm.url_open", icon="URL")
        open_store.url = "https://makertales.gumroad.com/"

        render_thumbs = row.operator(
            "render.render_gear_thumbnails", icon="FILE_REFRESH", text=""
        )

        if apply_filter:
            row = box.row()
            row.prop(prop_grp, "preset_filter", icon="VIEWZOOM", text="")

        row = box.row()
        row.template_icon_view(prop_grp, "preset_thumbnail", show_labels=True, scale=7)
        row = box.row()
        if prop_grp.thumb_rendering:
            row.label(text="Thumbnail Rendering in Progress, please wait.")
        row = box.row()
        apply_op = row.operator("object.apply_gear_preset", text="Apply Preset")

        box.label(text="Preset Limits:")
        box_col = box.column_flow()
        for i, scope_name in enumerate(cls._preset_scope_props):
            box_col.prop(prop_grp, scope_name)

    @classmethod
    def draw_header(cls, layout: UILayout, prop_grp: PrecisionGearsProps):
        row = layout.row()
        row.prop(prop_grp, "gear_type", text="Type")
        row.enabled = not prop_grp.rigging_enabled
        row = layout.row()

        cls._draw_presets(layout, prop_grp)

        # Draw presets
        row = layout.row(align=True)
        # row.prop(prop_grp, "preset", text="Preset")
        # row.operator(
        #     operators.OBJECT_OT_save_gear_preset.bl_idname, text="", icon="FILE_TICK"
        # )
        # row.operator(
        #     operators.OBJECT_OT_delete_gear_preset.bl_idname, text="", icon="TRASH"
        # )

    @classmethod
    def draw_prop_set(
        cls, layout, prop_set: dict, prop_grp: PrecisionGearsProps, label=""
    ):
        if prop_set is None:
            return None

        row = layout.row()
        box = row.box()
        col = box.column(align=True)
        if label != "":
            col.label(text=label)
        for key, prop_mapping in prop_set.items():
            if prop_mapping.newline:
                row = col.row()
            row.prop(prop_grp, key, text=prop_mapping.name)

    @classmethod
    def draw_bore_props(cls, layout, prop_grp: PrecisionGearsProps):
        if cls.bore_props is None:
            print("not bore")
            return None

        # print(prop_grp)
        row = layout.row()
        box = row.box()
        col = box.column(align=True)
        col.prop(prop_grp, "bore", text="Type")

        if prop_grp.bore == "NONE":
            return None

        bore_type = bores.get(prop_grp.bore)
        for key, prop_mapping in cls.bore_props.items():
            prop_name = bore_type.prop_names.get(key)
            if prop_name is not None:
                if prop_mapping.newline:
                    row = col.row()
                row.prop(prop_grp, key, text=prop_name)

        col.prop(prop_grp, "preview_bore", text="Preview")

    @classmethod
    def draw_rigging_props(cls, layout: UILayout, prop_grp: PrecisionGearsProps):
        row = layout.row()
        box = row.box()
        col = box.column(align=True)
        row = col.row()

        col.prop(prop_grp, "driver")

        row = col.row(align=True)
        row.prop(prop_grp, "rigging_mode")
        row = col.row(align=True)

        has_driver = prop_grp.driver is not None

        if not prop_grp.rigging_enabled:
            return None

        rig_tree = prop_grp.rig_tree
        if rig_tree is not None:
            root_driver_node = rig_tree.nodes.get(consts.ROOT_DRIVER_NODE_NAME)
            if root_driver_node is None:
                return None
        else:
            return None

        tree = root_driver_node.id_data
        state_controller = rigging.get_state_controller(tree)
        if state_controller is not None:
            col.prop(state_controller.inputs["State"], "default_value", text="State")
            row = col.row()
            row.prop(state_controller.inputs["Use Time"], "default_value", text="Use Time")
            row.prop( state_controller.inputs["State Length"], "default_value", text="State Length")

        row = col.row(align=True)
        if has_driver:
            op = row.operator(
                operators.OBJECT_OT_make_gear_compatible.bl_idname, text="Match Driver"
            )
            op.recursive = False
            op = row.operator(
                operators.OBJECT_OT_make_gear_compatible.bl_idname, text="Recursive"
            )
            op.recursive = True

        # TODO: Add ui prop groups per rigging mode
        rigging_node = prop_grp.rig_tree.nodes.get(prop_grp.rigging_node)

        if rigging_node is not None:
            rig_node_inputs = rigging_node.inputs
            if rigging_node.node_tree.name == "ROTATION":
                col.prop(
                    rig_node_inputs["Rot Offset"], "default_value", text="Rot Offset"
                )
                col.prop(rig_node_inputs["Angle"], "default_value", text="Angle")
                col.prop(
                    rig_node_inputs["Distance Offset"],
                    "default_value",
                    text="Distance Offset",
                )
                col.prop(
                    rig_node_inputs["Tangent Offset"],
                    "default_value",
                    text="Tangent Offset",
                )
                col.prop(
                    rig_node_inputs["Parallel Offset"],
                    "default_value",
                    text="Parallel Offset",
                )
                # col.prop(rig_node_inputs["Pitch Offset"], "default_value", text="Pitch Offset")
                # col.prop(rig_node_inputs["Parallel Offset"], "default_value", text="Parallel Offset")
            elif rigging_node.node_tree.name == "COMPOUND":
                col.prop(rig_node_inputs["Offset"], "default_value", text="Offset")
                col.prop(
                    rig_node_inputs["Rot Offset"], "default_value", text="Rot Offset"
                )
                col.prop(rig_node_inputs["Flip Z"], "default_value", text="Flip")
            elif rigging_node.node_tree.name == "BEVEL":
                col.prop(rig_node_inputs["Angle"], "default_value", text="Angle")
                col.prop(rig_node_inputs["Rot Offset"], "default_value", text="Rot Offset")
                col.prop(rig_node_inputs["Driver Apex Offset"], "default_value", text="Driver Apex Offset")
                col.prop(rig_node_inputs["Apex Offset"], "default_value", text="Apex Offset")
                # col.prop(rig_node_inputs["Offset"], "default_value", text="Offset")
            elif rigging_node.node_tree.name == "LINEAR":
                col.prop(rig_node_inputs["Angle"], "default_value", text="Angle")
                col.prop(rig_node_inputs["Offset"], "default_value", text="Offset")
                col.prop(
                    rig_node_inputs["Distance Offset"],
                    "default_value",
                    text="Distance Offset",
                )
                col.prop(
                    rig_node_inputs["Tangent Offset"],
                    "default_value",
                    text="Tangent Offset",
                )
                col.prop(
                    rig_node_inputs["Parallel Offset"],
                    "default_value",
                    text="Parallel Offset",
                )

            if has_driver:
                row = col.row(align=False)
                intersect_op = row.operator(
                    operators.OBJECT_OT_solve_gear_intersection.bl_idname,
                    text="Attempt Intersect Solve",
                )
                row = col.row(align=False)
                row.prop(prop_grp, "intersect_samples", text="Samples", emboss=False)
                row.prop(prop_grp, "intersect_solve_recursive", text="Recurse")
                row.prop(prop_grp, "intersect_solve_from_root", text="All")

                intersect_op.samples = prop_grp.intersect_samples
                intersect_op.recursive = prop_grp.intersect_solve_recursive
                intersect_op.from_root = prop_grp.intersect_solve_from_root

        col.operator(operators.OBJECT_OT_refresh_gear_rig.bl_idname)
        try:
            tree_name = tree.name
            rig_node_name = rigging_node.name
            rig_host_name = prop_grp.rig_host.name
            get_cursor_op = col.operator(operators.OBJECT_OT_cursor_to_rigged_gear.bl_idname)
            get_cursor_op.rig_host_name = rig_host_name
            get_cursor_op.tree_name = tree_name
            get_cursor_op.rig_node_name = rig_node_name
        except:
            pass

        # if has_driver:  # if prop_grp.root_driver is not None
        #     try:
        #         if prop_grp.rigging_mode == "MESHING":
        #             col.prop(prop_grp, "driver_offset")

        #         col.prop(prop_grp, "parallel_offset")
        #         driver_props = getattr(prop_grp.root_driver, GEAR_PROPS_ALIAS)
        #         col.prop(driver_props, "chain_actuation", text="Root Actuation")

        #         row = col.row(align=True)

        #         fix_intersect_op = operators.OBJECT_OT_solve_gear_intersection.bl_idname

        #         op = row.operator(fix_intersect_op, text="Set Offset")
        #         op.recursive = False
        #         op.mode = prop_grp.rigging_mode
        #         op.reverse_parallel = prop_grp.reverse_parallel_solve
        #         op = row.operator(fix_intersect_op, text="Recursive")
        #         op.recursive = True
        #         op.mode = prop_grp.rigging_mode
        #         op.reverse_parallel = prop_grp.reverse_parallel_solve

        #         row = col.row(align=True)
        #         if prop_grp.rigging_mode == "PARALLEL":
        #             # row.prop(prop_grp, "intersect_samples", text="Samples")
        #             row.prop(prop_grp, "reverse_parallel_solve", text="Flip Parallel")
        #         if prop_grp.rigging_mode == "MESHING":
        #             row.prop(prop_grp, "intersect_samples", text="Samples")
        #     except AttributeError:
        #         print("Invalid driver settings")
        #         pass
        # else:
        #     col.prop(prop_grp, "chain_actuation", text="Self Actuation")

    @classmethod
    def draw_tooth_modifier_props(cls, layout, prop_grp: PrecisionGearsProps):
        if cls.teeth_modification_props is None:
            return None

        row = layout.row()
        box = row.box()
        col = box.column(align=True)

        for key, prop_mapping in cls.teeth_modification_props.items():
            if not key.startswith("flank"):
                col.prop(prop_grp, key, text=prop_mapping.name)

        if "flank_modification" not in cls.teeth_modification_props.keys():
            return None

        if prop_grp.tooth_taper != 0:
            col.label(text="Tapering has limited compatibility with flank modification")

        col.prop(prop_grp, "flank_modification", text="Flank")

        if prop_grp.flank_modification == "NONE":
            return None

        # bore_type = bores.get(prop_grp.bore)
        modifier_type = flank_mods.get(prop_grp.flank_modification)

        for key, prop_mapping in cls.teeth_modification_props.items():
            prop_name = modifier_type.prop_names.get(key)
            if prop_name is not None:
                col.prop(prop_grp, key, text=prop_name)

    def set_prop_defaults(self):
        """
        Set the values in prop group based on cls.properties
        """
        # defaults_file_values = self.read_defaults
        defaults = presets.get_default_preset(self.gear_type)
        if defaults is None:
            return None
        self.props.editing = True
        try:
            for prop_name, value in defaults.items():
                try:
                    gear_prop = self.get_prop_map().get(prop_name, None)
                    if gear_prop is not None:
                        setattr(self.props, prop_name, value)
                except Exception as e:
                    # print(e)
                    continue
        except Exception as e:
            pass
            # print(e)
        finally:
            self.props.editing = False

    def create_gear(self) -> None:
        # Reset Error State
        self.error_encountered = False

        builder = self.builder(self.props)
        gear_mesh = builder.create_gear()
        if gear_mesh is None:
            self.error_encountered = True
        else:
            self.mesh = gear_mesh

    def center_gear(self) -> None:
        bbox = minify_bbox(get_bounding_box(self.mesh))
        center = mean_vectors(bbox)
        center *= self.centering_scaler
        for v in self.mesh.vertices:
            v.co -= center

    def add_set_screw(self) -> None:
        rotations = SET_SCREW_ARRANGEMENTS[self.props.set_screw - 1]
        bbox = minify_bbox(math_funcs.get_bounding_box(self.mesh))
        cutter_length = abs(bbox[0].x - bbox[1].x)
        radius = self.props.set_screw_radius
        resolution = self.props.set_screw_resolution
        set_screw_holes = mesh_generation.create_set_screw_holes(
            radius,
            rotations,
            resolution=resolution,
            depth=cutter_length,
            z_rot=self.props.set_screw_angle,
        )
        cut_offset = Vector((0, 0, self.props.set_screw_offset))

        if self.props.preview_set_screw:
            result = set_screw_holes
            bpy.data.meshes.remove(self.mesh)
        else:
            result = boolean_mesh(
                self.mesh, set_screw_holes, "DIFFERENCE", location=cut_offset
            )

        self.mesh = result

    def crease_angles(self) -> None:
        angle_crease_mesh(self.mesh, crease=self.props.crease)

    def add_extrusion(self, gear_bbox) -> None:
        radius = self.props.extrusion_radius
        min_z = gear_bbox[0].z
        max_z = gear_bbox[1].z

        bool_buffer = self.props.extrusion_size * 0.01
        depth = self.props.extrusion_size + bool_buffer
        location = Vector((0, 0, max_z - bool_buffer))
        resolution = self.props.extrusion_resolution
        extrusion = mesh_generation.new_cylinder_mesh(radius, resolution, depth=depth)
        floor_mesh(extrusion)

        if self.props.extrusion_top:
            self.mesh = boolean_mesh(
                self.mesh,
                extrusion,
                "UNION",
                location=location,
                solver=self.props.bool_solver,
            )
        if self.props.extrusion_bottom:
            mirror_z = min_z - depth + bool_buffer
            mirror_loc = Vector((0, 0, mirror_z))
            self.mesh = boolean_mesh(
                self.mesh,
                extrusion,
                "UNION",
                location=mirror_loc,
                solver=self.props.bool_solver,
            )

    def add_bore(self) -> Mesh:
        """Add bore to mesh, return modified copy"""
        props = self.props

        gear_bbox = minify_bbox(get_bounding_box(self.mesh))
        min_z = gear_bbox[0].z
        max_z = gear_bbox[1].z
        z_size = abs(min_z - max_z)
        origin_offset = (gear_bbox[0] + gear_bbox[1]) * 0.5
        origin_offset *= Vector((0, 0, 1))

        bore_creator = bores.get(props.bore)
        bore = bore_creator(
            props.bore_size,
            z_size * 1.2,
            props.bore_param_a,
            props.bore_param_b,
            props.bore_resolution,
            props.bore_subtype,
        )

        # Center bore
        translate_mesh(bore.mesh, origin_offset)

        if self.props.preview_bore:
            result = bore.mesh
            bpy.data.meshes.remove(self.mesh)
        else:
            result = boolean_mesh(
                self.mesh, bore.mesh, "DIFFERENCE", solver=self.props.bool_solver
            )
            # Cleanup
            bore.clear()

        self.mesh = result

    @abstractmethod
    def calculate_attributes(self) -> Dict[str, float]:
        """Calculate gear attributes and return as dict for use as a property"""
        raise NotImplementedError


@dataclass
class RackGear(GearCreator):
    props: PrecisionGearsProps
    gear_type = "RACK"
    builder = gear_builders.InvoluteGearRack

    gear_props = {
        "teeth": PropertyMapping("Teeth", default=15),
        "module": PropertyMapping("Module", min_val=0.0001, default=0.1),
        "height": PropertyMapping("Height", default=0.5),
        "width": PropertyMapping("Width", default=0.5),
        "pressure_angle": PropertyMapping("Pressure Angle", default=radians(20)),
        "head": PropertyMapping("Head", default=0),
        "helix_angle": PropertyMapping("Helix Angle", default=0),
        "z_resolution": PropertyMapping("Z Resolution", default=10),
        "clearance": PropertyMapping("Clearance", default=0.25),
        "herringbone": PropertyMapping("Herringbone", default=False, newline=False),
    }

    teeth_modification_props = {
        "tooth_taper": PropertyMapping("Taper", default=0),
        "tooth_taper_offset": PropertyMapping("Taper Offset", default=0),
        "tip_relief": PropertyMapping("Tip Relief", default=0),
        "tip_relief_resolution": PropertyMapping("Tip Relief Resolution", default=2),
        "root_relief": PropertyMapping("Root Relief", default=0),
        "root_relief_resolution": PropertyMapping("Root Relief Resolution", default=2),
        "flank_modification": PropertyMapping("Flank Mod", default="NONE"),
        "flank_mod_param_a": PropertyMapping("Param A", default=0),
        "flank_mod_param_b": PropertyMapping("Param B", default=0),
    }

    extra_props = {
        "do_crease": PropertyMapping("Do Crease", default=True),
        "crease": PropertyMapping("Angle", default=radians(35)),
        "add_weld": PropertyMapping("Weld", default=True),
        "weld_dist": PropertyMapping("Distance", default=0.001),
        "scaler": PropertyMapping("Scale", default=1),
        "unwrap": PropertyMapping("Unwrap", default=True),
        "uv_seam_angle": PropertyMapping("Seam Angle", default=radians(45)),
        # "smooth_angle": PropertyMapping("Smooth Angle", default=radians(30)),
    }

    toggle_props = {
        # "bore_enabled": PropertyMapping("Bore Enabled", default=False),
        "tooth_mod_enabled": PropertyMapping("Tooth Mod Enabled", default=False),
        # "set_screw_enabled": PropertyMapping("Set Screw Enabled", default=False),
        # "extrusion_enabled": PropertyMapping("Extrusion Enabled", default=False),
    }

    def calculate_attributes(self) -> Dict[str, float]:
        module = self.props.module
        width = self.props.width
        teeth = self.props.teeth
        pitch_dist = (((module * 2.25) + width) / 2) - module
        # pitch_dist += self.props.pitch_offset
        return {
            "Addendum": module,  # The distance between reference line and tooth tip
            "Dedendum": module * 1.25,  # The distance between the reference line and tooth root
            "Tooth Depth": 2.25 * module,  # The distance between the tooth rip and tooth root
            "Working Depth": 2.0 * module,  # Depth of teooth meshed with mating gear
            "Pitch": module * pi,  # Distance between corresponding points on adjacent teeth
            "Rack Pitch Distance": pitch_dist,
            # "Pitch Length": module * pi * teeth,
            "Pitch Length": module * pi * teeth,
        }


@dataclass
class SpurGear(GearCreator):
    props: PrecisionGearsProps
    gear_type = "SPUR"
    builder = gear_builders.InvoluteGear
    centering_scaler = Vector((0, 0, 1))

    gear_props = {
        "teeth": PropertyMapping("Teeth", default=15),
        "module": PropertyMapping("Module", min_val=0, default=0.1),
        "height": PropertyMapping("Height", default=0.5),
        "pressure_angle": PropertyMapping("Pressure Angle", default=radians(20)),
        "complexity": PropertyMapping("Complexity", default=10),
        "shift": PropertyMapping("Shift", default=0),
        "head": PropertyMapping("Head", default=0.0),
        "backlash": PropertyMapping("Backlash", default=0.0),
        "helix_angle": PropertyMapping("Helix Angle", default=0),
        "z_resolution": PropertyMapping("Z Resolution", default=10),
        "clearance": PropertyMapping("Clearance", default=0.25),
        "herringbone": PropertyMapping("Herringbone", default=False),
        "undercut": PropertyMapping("Undercut", default=False, newline=False),
    }

    teeth_modification_props = {
        "tooth_taper": PropertyMapping("Taper", default=0),
        "tooth_taper_offset": PropertyMapping("Taper Offset", default=0),
        "tip_relief": PropertyMapping("Tip Relief", default=0),
        "tip_relief_resolution": PropertyMapping("Tip Relief Resolution", default=2),
        "worm_cut": PropertyMapping("Worm Cut", default=0),
        "worm_cut_scale": PropertyMapping("Worm Cut Scale", default=1),
        "flank_modification": PropertyMapping("Flank Mod", default="NONE"),
        "flank_mod_param_a": PropertyMapping("Param A", default=0),
        "flank_mod_param_b": PropertyMapping("Param B", default=0),
    }

    bore_props = {
        "bore": PropertyMapping("Type", default="ROUND_SUBBED"),
        "bore_size": PropertyMapping("Bore Size", default=0.3, min_val=0),
        "bore_param_a": PropertyMapping("Param A", default=1.0),
        "bore_param_b": PropertyMapping("Param B", default=1.55),
        "bore_resolution": PropertyMapping("Subtype", default=64),
        "bore_subtype": PropertyMapping("Subtype", default=True),
        "preview_bore": PropertyMapping("Preview", default=False),
    }

    extrusion_props = {
        "extrusion_size": PropertyMapping("Size", default=0.45),
        "extrusion_radius": PropertyMapping("Radius", default=0.5),
        "extrusion_resolution": PropertyMapping("Resolution", default=64),
        "extrusion_top": PropertyMapping("Top", default=True),
        "extrusion_bottom": PropertyMapping("Bottom", default=False, newline=False),
    }

    set_screw_props = {
        "set_screw": PropertyMapping("Arrangement", default=0),
        "set_screw_offset": PropertyMapping("Offset", default=0),
        "set_screw_radius": PropertyMapping("Radius", default=0.1),
        "set_screw_resolution": PropertyMapping("Resolution", default=64),
        "set_screw_angle": PropertyMapping("Angle", default=0),
        "preview_set_screw": PropertyMapping("Preview", default=False),
    }

    extra_props = {
        "bool_solver": PropertyMapping("Boolean", default="EXACT"),
        "do_crease": PropertyMapping("Crease", default=True),
        "crease": PropertyMapping("", default=radians(35), newline=False),
        "add_weld": PropertyMapping("Weld", default=True),
        "weld_dist": PropertyMapping("", default=0.001, newline=False),
        "scaler": PropertyMapping("Scale", default=1),
        "unwrap": PropertyMapping("Unwrap", default=True),
        "uv_seam_angle": PropertyMapping("Seam Angle", default=radians(45)),
        # "smooth_angle": PropertyMapping("Smooth Angle", default=radians(30)),
    }

    toggle_props = {
        "bore_enabled": PropertyMapping("Bore Enabled", default=False),
        "tooth_mod_enabled": PropertyMapping("Tooth Mod Enabled", default=False),
        "set_screw_enabled": PropertyMapping("Set Screw Enabled", default=False),
        "extrusion_enabled": PropertyMapping("Extrusion Enabled", default=False),
    }

    def calculate_attributes(self) -> Dict[str, float]:
        module = self.props.module
        teeth = self.props.teeth
        return {
            "Addendum": module,  # Distance between reference line and tooth tip
            "Dedendum": module * 1.25,  # Distance between the reference line and tooth root
            "Tooth Depth": 2.25 * module,  # Distance between the tooth rip and tooth root
            "Working Depth": 2.0 * module,  # Depth of teeth meshed with mating gear
            "Pitch": module * pi,  # Distance between corrosponding points on adjacent teeth
            "Pitch Radius": ((module * teeth) / 2),
            "Pitch Circumference": (module * teeth) * pi,  # Length of pitch line
        }


@dataclass
class InternalGear(GearCreator):
    props: PrecisionGearsProps
    gear_type = "INTERNAL"
    builder = gear_builders.InvoluteGear
    centering_scaler = Vector((0, 0, 1))

    gear_props = {
        "teeth": PropertyMapping("Teeth", default=40),
        "module": PropertyMapping("Module", min_val=0, default=0.1),
        "height": PropertyMapping("Height", default=0.5),
        "width": PropertyMapping("Width", default=0.5),
        "pressure_angle": PropertyMapping("Pressure Angle", default=radians(20)),
        "complexity": PropertyMapping("Complexity", default=10),
        "shift": PropertyMapping("Shift", default=0.5),
        "head": PropertyMapping("Head", default=-0.78),
        "backlash": PropertyMapping("Backlash", default=0.04),
        "helix_angle": PropertyMapping("Helix Angle", default=0),
        "z_resolution": PropertyMapping("Helix Steps", default=10),
        "clearance": PropertyMapping("Clearance", default=0.25),
        "herringbone": PropertyMapping("Herringbone", default=False),
        "undercut": PropertyMapping("Undercut", default=False, newline=False),
    }

    teeth_modification_props = {
        "tooth_taper": PropertyMapping("Taper", default=0),
        "tooth_taper_offset": PropertyMapping("Taper Offset", default=0),
        "tip_relief": PropertyMapping("Tip Relief", default=0),
        "tip_relief_resolution": PropertyMapping("Tip Relief Resolution", default=2),
        "flank_modification": PropertyMapping("Flank Mod", default="NONE"),
        "flank_mod_param_a": PropertyMapping("Param A", default=0),
        "flank_mod_param_b": PropertyMapping("Param B", default=0),
    }

    extra_props = {
        "do_crease": PropertyMapping("Crease", default=True),
        "crease": PropertyMapping("", default=radians(35), newline=False),
        "add_weld": PropertyMapping("Weld", default=True),
        "weld_dist": PropertyMapping("", default=0.001, newline=False),
        "scaler": PropertyMapping("Scale", default=1),
        "unwrap": PropertyMapping("Unwrap", default=True),
        "uv_seam_angle": PropertyMapping("Seam Angle", default=radians(45)),
        # "smooth_angle": PropertyMapping("Smooth Angle", default=radians(30)),
    }

    toggle_props = {
        "tooth_mod_enabled": PropertyMapping("Tooth Mod Enabled", default=False),
    }

    def calculate_attributes(self) -> Dict[str, float]:
        attributes = {}
        module = self.props.module
        teeth = self.props.teeth
        pitch_radius = (module * teeth) / 2
        # pitch_radius -= self.props.pitch_offset
        return {
            "Addendum": module,  # The distance between reference line and tooth tip
            "Dedendum": module
            * 1.25,  # The distance between the reference line and tooth root
            "Tooth Depth": 2.25
            * module,  # The distance between the tooth rip and tooth root
            "Working Depth": 2.0 * module,  # Depth of teooth meshed with mating gear
            "Pitch": (module * teeth * pi) / teeth,
            "Pitch Radius": pitch_radius,
            "Pitch Circumference": (module * teeth) * pi,  # Length of pitch line
        }
        return attributes


@dataclass
class WormGear(GearCreator):
    props: PrecisionGearsProps
    gear_type = "WORM"
    builder = gear_builders.WormGear
    centering_scaler = Vector((0, 0, 1))

    gear_props = {
        "teeth": PropertyMapping("Teeth", default=3),
        "module": PropertyMapping("Module", min_val=0.001, default=0.1),
        "height": PropertyMapping("Height", default=2, min_val=0.001),
        "pressure_angle": PropertyMapping("Pressure Angle", default=radians(20)),
        "diameter": PropertyMapping("Diameter", default=0.5),
        "head": PropertyMapping("Head", default=0.0),
        "z_resolution": PropertyMapping("Rotation Steps", default=32),
        "clearance": PropertyMapping("Clearance", default=0.0),
        "reverse_pitch": PropertyMapping("Reverse Pitch", default=False),
    }

    bore_props = {
        "bore": PropertyMapping("Bore", default="NONE"),
        "bore_size": PropertyMapping("Bore Size", default=0.3, min_val=0),
        "bore_param_a": PropertyMapping("Param A", default=1.0),
        "bore_param_b": PropertyMapping("Param B", default=1.55),
        "bore_resolution": PropertyMapping("Subtype", default=64),
        "bore_subtype": PropertyMapping("Subtype", default=True),
        "preview_bore": PropertyMapping("Preview", default=False),
    }

    extrusion_props = {
        "extrusion_size": PropertyMapping("Size", default=0.32),
        "extrusion_radius": PropertyMapping("Radius", default=0.35),
        "extrusion_top": PropertyMapping("Top", default=True),
        "extrusion_bottom": PropertyMapping("Bottom", default=True, newline=False),
        "extrusion_resolution": PropertyMapping("Resolution", default=64),
    }

    set_screw_props = {
        "set_screw": PropertyMapping("Set Screw", default=0),
        "set_screw_offset": PropertyMapping("Offset", default=0),
        "set_screw_radius": PropertyMapping("Radius", default=0.1),
        "set_screw_resolution": PropertyMapping("Resolution", default=64),
        "set_screw_angle": PropertyMapping("Angle Offset", default=0),
        "preview_set_screw": PropertyMapping("Preview", default=False, newline=False),
    }

    extra_props = {
        "bool_solver": PropertyMapping("Boolean", default="EXACT"),
        "do_crease": PropertyMapping("Crease", default=True),
        "crease": PropertyMapping("", default=radians(35), newline=False),
        "add_weld": PropertyMapping("Weld", default=True),
        "weld_dist": PropertyMapping("", default=0.001, newline=False),
        "scaler": PropertyMapping("Scale", default=1),
        "unwrap": PropertyMapping("Unwrap", default=True),
        "uv_seam_angle": PropertyMapping("Seam Angle", default=radians(45)),
        # "smooth_angle": PropertyMapping("Smooth Angle", default=radians(30)),
    }

    toggle_props = {
        "bore_enabled": PropertyMapping("Bore Enabled", default=False),
        "set_screw_enabled": PropertyMapping("Set Screw Enabled", default=False),
        "extrusion_enabled": PropertyMapping("Extrusion Enabled", default=False),
    }

    def calculate_attributes(self) -> Dict[str, float]:
        module = self.props.module
        pitch_radius = self.props.diameter / 2
        return {
            "Addendum": module,  # The distance between reference line and tooth tip
            "Dedendum": module
            * 1.25,  # The distance between the reference line and tooth root
            "Tooth Depth": 2.25
            * module,  # The distance between the tooth rip and tooth root
            "Working Depth": 2.0 * module,  # Depth of teooth meshed with mating gear
            # "Pitch": module * pi,  # Distance between corresponding points on adjacent teeth
            "Pitch Radius": pitch_radius,  # Distance between corresponding points on adjacent teeth
            # "Pitch Length": module * pi * teeth,
        }


@dataclass
class BevelGear(GearCreator):
    props: PrecisionGearsProps
    gear_type = "BEVEL"
    builder = gear_builders.BevelGear
    centering_scaler = Vector((0, 0, 1))
    gear_props = {
        "teeth": PropertyMapping("Teeth", default=20, min_val=1),
        "module": PropertyMapping("Module", min_val=0.000001, default=0.1),
        "height": PropertyMapping("Length", default=0.6),
        "pressure_angle": PropertyMapping(
            "Pressure Angle", default=radians(15), min_val=0, max_val=radians(30)
        ),
        "pitch_angle": PropertyMapping(
            "Pitch Angle", min_val=0.000001, default=radians(45)
        ),
        "backlash": PropertyMapping("Backlash", default=0.0),
        "complexity": PropertyMapping("Complexity", min_val=2, default=6),
        "helix_angle": PropertyMapping("Helix Angle", default=0),
        "z_resolution": PropertyMapping("Helix Steps", default=10),
        "clearance": PropertyMapping("Tooth Scaler", default=0.8),
    }

    bore_props = {
        "bore": PropertyMapping("Bore", default="CROSS"),
        # "bore_size": PropertyMapping( "Bore Size", default=0.14, min_val=0, max_val=len(bores.bores)),
        "bore_size": PropertyMapping("Bore Size", default=0.14, min_val=0),
        "bore_param_a": PropertyMapping("Param A", default=1.0),
        "bore_param_b": PropertyMapping("Param B", default=1.0),
        "bore_resolution": PropertyMapping("Subtype", default=64),
        "bore_subtype": PropertyMapping("Subtype", default=True),
        "preview_bore": PropertyMapping("Preview", default=False, newline=False),
    }

    extrusion_props = {
        "extrusion_size": PropertyMapping("Size", default=0.45),
        "extrusion_radius": PropertyMapping("Radius", default=0.5),
        "extrusion_resolution": PropertyMapping("Resolution", default=64),
        "extrusion_top": PropertyMapping("Top", default=False),
        "extrusion_bottom": PropertyMapping("Bottom", default=True, newline=False),
    }

    set_screw_props = {
        "set_screw": PropertyMapping("Arrangement", default=0),
        "set_screw_offset": PropertyMapping("Offset", default=0),
        "set_screw_radius": PropertyMapping("Radius", default=0.1),
        "set_screw_resolution": PropertyMapping("Resolution", default=64),
        "set_screw_angle": PropertyMapping("Angle", default=0),
        "preview_set_screw": PropertyMapping("Preview", default=False),
    }

    extra_props = {
        "bool_solver": PropertyMapping("Boolean", default="EXACT"),
        "do_crease": PropertyMapping("Crease", default=True),
        "crease": PropertyMapping("", default=radians(35), newline=False),
        "add_weld": PropertyMapping("Weld", default=True),
        "weld_dist": PropertyMapping("", default=0.001, newline=False),
        "scaler": PropertyMapping("Scale", default=1),
        "unwrap": PropertyMapping("Unwrap", default=True),
        "uv_seam_angle": PropertyMapping("Seam Angle", default=radians(45)),
        # "smooth_angle": PropertyMapping("Smooth Angle", default=radians(30)),
    }

    toggle_props = {
        "bore_enabled": PropertyMapping("Bore Enabled", default=False),
        # "tooth_mod_enabled": PropertyMapping("Tooth Mod Enabled", default=False),
        "set_screw_enabled": PropertyMapping("Set Screw Enabled", default=False),
        "extrusion_enabled": PropertyMapping("Extrusion Enabled", default=False),
    }

    def add_extrusion(self, gear_bbox) -> None:
        radius = self.props.extrusion_radius
        min_z = gear_bbox[0].z
        max_z = gear_bbox[1].z

        bool_buffer = self.props.extrusion_size * 0.01
        depth = self.props.extrusion_size + bool_buffer
        location = Vector((0, 0, max_z - bool_buffer))
        resolution = self.props.extrusion_resolution
        extrusion = mesh_generation.new_cylinder_mesh(radius, resolution, depth=depth)

        floor_mesh(extrusion)

        # Find top face distance
        bm = bmesh.new()
        bm.from_mesh(self.mesh)
        tree = bvhtree.BVHTree.FromBMesh(bm)
        bm.free()
        origin = Vector((0, 0, max_z))
        neg_z = Vector((0, 0, -1))
        cast = tree.ray_cast(origin, neg_z)
        distance = cast[-1]
        location.z -= distance

        if self.props.extrusion_top:
            self.mesh = boolean_mesh(
                self.mesh,
                extrusion,
                "UNION",
                location=location,
                solver=self.props.bool_solver,
            )
        if self.props.extrusion_bottom:
            mirror_z = min_z - depth + bool_buffer
            mirror_loc = Vector((0, 0, mirror_z))
            self.mesh = boolean_mesh(
                self.mesh,
                extrusion,
                "UNION",
                location=mirror_loc,
                solver=self.props.bool_solver,
            )

    def calculate_attributes(self) -> Dict[str, float]:
        attributes = {}
        module = self.props.module
        teeth = self.props.teeth
        dedendum = module * 1.25
        # pitch_offset = self.props.pitch_offset
        pitch_radius = (teeth * module) / 2
        # pitch_radius += self.props.pitch_offset
        apex = pitch_radius / math.tan(self.props.pitch_angle)
        attributes["pitch_radius"] = pitch_radius
        attributes["pitch_circumference"] = (module * teeth) * pi
        attributes["apex"] = apex
        return attributes


@dataclass
class GT2Gear(GearCreator):
    props: PrecisionGearsProps
    gear_type = "GT2GEAR"
    builder = gear_builders.GT2Gear
    centering_scaler = Vector((0, 0, 1))

    gear_props = {
        "teeth": PropertyMapping("Teeth", default=15),
        "height": PropertyMapping("Height", default=0.5),
        "complexity": PropertyMapping("Complexity", min_val=2, default=5),
        "head": PropertyMapping("Boundary Width", min_val=0, default=0.4),
        "clearance": PropertyMapping("Boundary Offset", default=0.00),
    }

    bore_props = {
        "bore": PropertyMapping("Type", default="ROUND_SUBBED"),
        "bore_size": PropertyMapping("Bore Size", default=0.3, min_val=0),
        "bore_param_a": PropertyMapping("Param A", default=1.0),
        "bore_param_b": PropertyMapping("Param B", default=1.55),
        "bore_resolution": PropertyMapping("Subtype", default=64),
        "bore_subtype": PropertyMapping("Subtype", default=True),
        "preview_bore": PropertyMapping("Preview", default=False),
    }

    extrusion_props = {
        "extrusion_size": PropertyMapping("Size", default=0.45),
        "extrusion_radius": PropertyMapping("Radius", default=0.5),
        "extrusion_resolution": PropertyMapping("Resolution", default=64),
        "extrusion_top": PropertyMapping("Top", default=True),
        "extrusion_bottom": PropertyMapping("Bottom", default=False, newline=False),
    }

    set_screw_props = {
        "set_screw": PropertyMapping("Arrangement", default=0),
        "set_screw_offset": PropertyMapping("Offset", default=0),
        "set_screw_radius": PropertyMapping("Radius", default=0.1),
        "set_screw_resolution": PropertyMapping("Resolution", default=64),
        "set_screw_angle": PropertyMapping("Angle", default=0),
        "preview_set_screw": PropertyMapping("Preview", default=False),
    }

    extra_props = {
        "bool_solver": PropertyMapping("Boolean", default="EXACT"),
        "do_crease": PropertyMapping("Crease", default=True),
        "crease": PropertyMapping("", default=radians(35), newline=False),
        "add_weld": PropertyMapping("Weld", default=True),
        "weld_dist": PropertyMapping("", default=0.001, newline=False),
        "scaler": PropertyMapping("Scale", default=1),
        "unwrap": PropertyMapping("Unwrap", default=True),
        "uv_seam_angle": PropertyMapping("Seam Angle", default=radians(45)),
        # "smooth_angle": PropertyMapping("Smooth Angle", default=radians(30)),
    }

    toggle_props = {
        "bore_enabled": PropertyMapping("Bore Enabled", default=False),
        "tooth_mod_enabled": PropertyMapping("Tooth Mod Enabled", default=False),
        "set_screw_enabled": PropertyMapping("Set Screw Enabled", default=False),
        "extrusion_enabled": PropertyMapping("Extrusion Enabled", default=False),
    }

    def calculate_attributes(self) -> Dict[str, float]:
        # pitch_offset = self.props.pitch_offset
        attributes = {
            "radius": 0.0,
            "rig_ratio": 0.0,
            "init_loc": (0.0, 0.0, 0.0),
        }
        return attributes


def get_gear_definition(gear_type: str):
    builders = GearCreator.__subclasses__()
    known_types = []
    for builder in builders:
        known_types.append(builder.gear_type)
        if builder.gear_type == gear_type:
            return builder
    else:
        error = f"Cannot find a builder for {gear_type} "
        details = f"Recognized types are {known_types}"
        raise ValueError("".join((error, details)))
