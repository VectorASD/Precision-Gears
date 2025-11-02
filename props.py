from __future__ import annotations
from itertools import count
import logging
from math import radians
from typing import Set

import bmesh
import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    IntProperty,
    FloatProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import (
    AddonPreferences,
    Object,
    PropertyGroup,
    Context,
    GeometryNodeTree,
    Mesh,
    GeometryNode,
    GeometryNodeGroup,
)
import bpy.utils.previews

from .preset_prop_groups import PRESET_PROP_GROUPS
from .rigging import update_rigging
from . import (
    consts,
    gear_definitions ,
    bores,
    presets,
    mesh_editing,
)

preview_collections = {}
global_thumb_ref = set()


# def update_shade_smooth(self, context: Context):
#     target = context.active_object
#     if consts.BPY_4_1_COMPAT_REQUIRED:
#         bpy.ops.object.shade_smooth_by_angle(angle=self.smooth_angle, keep_sharp_edges=False)
#     else:
#         target.data.use_auto_smooth = True
#         target.data.auto_smooth_angle = self.smooth_angle

def uv_prop_callback(self: PrecisionGearsProps, ctx: Context):
    if self.unwrap:
        gear: Object = ctx.active_object
        mesh: Mesh = gear.data
        if mesh.is_editmode:
            bm = bmesh.from_edit_mesh(mesh)
        else:
            bm = bmesh.new()
            bm.from_mesh(mesh)
        
        mesh_editing.clear_seams(bm)
        mesh_editing.add_seams_by_angle(bm, self.uv_seam_angle)

        if mesh.is_editmode:
            bmesh.update_edit_mesh(mesh)
        else:
            bm.to_mesh(mesh)
            bm.free()
        
        # Make gear active
        gear.select_set(True)
        ctx.view_layer.objects.active = gear
        ctx.view_layer.update()

        init_mode = gear.mode

        if not mesh.uv_layers:
            mesh.uv_layers.new()

        # override = ctx.copy()
        # override["active_object"] = gear
        # override["objects_in_mode"] = [gear,]
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        ret = bpy.ops.uv.unwrap(margin=0.001)
        # print(ret)
        bpy.ops.object.mode_set(mode="OBJECT")

    # mesh.is_editmode


def rigging_enum(self: PrecisionGearsProps, ctx: Context):
    index = count()
    enum = []
    if self.gear_type in ["WORM", "SPUR", "INTERNAL"]:
        enum.append(("ROTATION", "Rotation", "Rotation", next(index)))
    if self.gear_type == "BEVEL":
        enum.append(("BEVEL", "Bevel", "Bevel", next(index)))
    if self.gear_type == "RACK":
        enum.append(("LINEAR", "Linear", "Linear", next(index)))
    enum.append(("COMPOUND", "Compound", "Compound", next(index)))
    return enum


def populate_thumb_enum(self: PrecisionGearsProps, context: Context):
    """Callback for enum of png images for thumbnail preview collection"""
    enum_items = []
    props = self

    # Used for populating preset filter enum.
    if not isinstance(props, PrecisionGearsProps):
        props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)

    preview_collection = preview_collections[consts.THUMB_COLLECTION_ALIAS]

    if context is None:
        return enum_items

    # preset_category = props.preset_category
    preset_dir = consts.PRESETS_DIR / props.gear_type
    n_thumbs = len(list(preset_dir.glob(f"*{consts.PRESET_THUMB_SUFFIX}")))
    filter_is_applied = all((props.apply_preset_filter, props.preset_filter != "",))
    filter_has_changed = preview_collection.active_filter != props.preset_filter
    # preset_dir_changed =  preset_dir != preview_collection.active_directory
    # thumb_count_changed = n_thumbs != preview_collection.n_known_thumbs

    update_conditions = {
        "new preset dir": preset_dir != preview_collection.active_directory,
        "thumb count change": n_thumbs != preview_collection.n_known_thumbs,
        "filter string change": all((filter_is_applied, filter_has_changed)),
        "forced update": preview_collection.force_update,
    }

    changes = []
    for test, condition in update_conditions.items():
        if condition:
            changes.append(test)
        
    if not changes or props.thumb_rendering:
        return preview_collection.previews
    
    preview_collection.active_directory = preset_dir
    preview_collection.active_filter = props.preset_filter

    # If preset_dir_changed or thumb_count_changed:
    try:
        preview_collection.clear()
    except ResourceWarning:
        pass

    gear_presets = presets.get_presets(self.gear_type)
    n_known_thumbs = 0

    if preset_dir.exists:
        for indx, preset in enumerate(gear_presets.values()):
            thumb_fname = f"{preset['thumb']}{consts.PRESET_THUMB_SUFFIX}"
            thumb_path = preset_dir / thumb_fname
            preset_name = preset["preset_name"]
            if not thumb_path.exists():
                print(preset_name, " has no thumb")
                continue

            n_known_thumbs += 1

            if filter_is_applied:
                if props.preset_filter not in preset_name:
                    continue

            value = preset["preset_name"]
            label = preset_name
            thumb_key = (
                f"{props.gear_type}_{preset['preset_name']}"
            )
            filepath = str(thumb_path)

            icon = preview_collection.get(thumb_key)
            if not icon:
                thumb = preview_collection.load(thumb_key, filepath, "IMAGE")
            else:
                thumb = preview_collection[thumb_key]

            item = (value, label, value, thumb.icon_id, indx)
            global_thumb_ref.add(item)
            enum_items.append(item)

    global_thumb_ref.union(enum_items)

    preview_collection.n_known_thumbs = n_known_thumbs
    preview_collection.previews = enum_items
    preview_collection.force_update = False
    preview_collection.active_directory = preset_dir
    return preview_collection.previews


def clear_preset_filter(self, context: Context) -> None:
    if not self.apply_preset_filter:
        preview_collection = preview_collections[consts.THUMB_COLLECTION_ALIAS]
        preview_collection.force_update = True


class PrecisionGearsProps(PropertyGroup):
    """
    Properties of active gear
    """

    def ui_rig_modes_enum(self, _):
        """ Return enum of rigging modes applicable to self.gear_type """
        compatible_drivers = consts.VALID_RIG_DRIVERS[self.gear_type]
        enum = []
        for i, mode in enumerate(compatible_drivers):
            label = mode.replace("_", " ").title()
            item = (mode, label, label, i)
            enum.append(item)
        return enum

    def load_presets(self, context):
        return presets.get_presets_enum(self.gear_type)
    
    def _get_preset_prop_exclusions(self) -> Set[str]:
        """
        Get a list of props to be excluded from preset application based on scope bools use_preset_*
        and return as a set
        """
        #  Loop over these pairs of conditions and proper name sets
        #   if true, union with prop_exclusion set
        prop_exclusions = set()
        exclusion_conditions = (
            (self.use_preset_base, PRESET_PROP_GROUPS["base"]),
            (self.use_preset_tooth_mod, PRESET_PROP_GROUPS["tooth_mod"]),
            (self.use_preset_bore, PRESET_PROP_GROUPS["bore"]),
            (self.use_preset_extrusion, PRESET_PROP_GROUPS["extrusion"]),
            (self.use_preset_set_screw, PRESET_PROP_GROUPS["set_screw"]),
            (self.use_preset_general, PRESET_PROP_GROUPS["general"]),
        )
        for is_enabled, exclusions in exclusion_conditions:
            if not is_enabled:
                prop_exclusions = prop_exclusions.union(exclusions)
        return prop_exclusions

    def apply_preset(self, context):
        """ Set the values in prop group based on cls.properties """
        if self.preset_thumbnail == "None":
            return None

        self.editing = True
        prop_exclusions = self._get_preset_prop_exclusions()
        # print(self.use_preset_base, 'teeth' in prop_exclusions)
        print(f"Applying preset: {self.preset_thumbnail}")
        presets.apply_preset(self, self.preset_thumbnail, prop_exclusions)
        self.editing = False
        gear_definitions.update_gear(self, context)

    is_gear: BoolProperty(name="Is Gear", default=False)
    # is_rigged_object: BoolProperty(name="Is Rigged Object", default=False)

    # States
    thumb_rendering: BoolProperty(default=False)
    editing: BoolProperty(name="Editing", default=False)

    gear_type: EnumProperty(name="Gear Type", items=consts.GEAR_TYPES_ENUM, update=gear_definitions.update_gear, default=0)
    # preset: EnumProperty(name="Preset", items=load_presets, default=0, update=apply_preset)
    preset_thumbnail: EnumProperty(items=populate_thumb_enum, default=0)

    # Preset filtering
    apply_preset_filter: BoolProperty(default=False, description="Preset Filter", update=clear_preset_filter)
    preset_filter: StringProperty(default="", description="Preset Filter")

    # Preset scope toggles
    # TODO: These should probably be an enum
    use_preset_base: BoolProperty(name="Base", default=True)
    use_preset_tooth_mod: BoolProperty(name="Tooth Mod", default=True)
    use_preset_bore: BoolProperty(name="Bore", default=True)
    use_preset_extrusion: BoolProperty(name="Extrusion", default=True)
    use_preset_set_screw: BoolProperty(name="Set Screw", default=True)
    use_preset_general: BoolProperty(name="General", default=False)

    # Base Gear Props
    herringbone: BoolProperty(name="Herringbone", default=False, update=gear_definitions.update_gear)
    module: FloatProperty(name="Module", update=gear_definitions.update_gear, subtype="DISTANCE", default=0.1, min=0.000001)
    helix_angle: FloatProperty(name="Helix Angle", update=gear_definitions.update_gear, subtype="ANGLE", default=0)
    z_resolution: IntProperty(name="Helix Steps", update=gear_definitions.update_gear, default=1, min=1, soft_max=20)
    pressure_angle: FloatProperty(name="Pressure Angle", update=gear_definitions.update_gear, subtype="ANGLE", default=radians(20), min=0.0001, soft_max=radians(30), max=radians(40),)
    pitch_angle: FloatProperty(name="Pitch Angle", update=gear_definitions.update_gear, subtype="ANGLE", default=radians(45), min=radians(0), max=radians(89.9),)
    length: FloatProperty(name="Length", update=gear_definitions.update_gear, subtype="DISTANCE", default=4, min=0)
    diameter: FloatProperty(name="Diameter", update=gear_definitions.update_gear, subtype="DISTANCE", default=4, min=0.00001)
    width: FloatProperty(name="Width", update=gear_definitions.update_gear, subtype="DISTANCE", default=4, min=0)
    backlash: FloatProperty(name="Backlash", update=gear_definitions.update_gear, default=0.0, min=0)
    height: FloatProperty(name="Height", update=gear_definitions.update_gear, subtype="DISTANCE", default=4, min=0)
    rotate: FloatProperty(name="Rotate", update=gear_definitions.update_gear, subtype="ANGLE", default=0)
    teeth: IntProperty(name="Teeth", update=gear_definitions.update_gear, min=1, default=15, soft_max=128)
    undercut: BoolProperty(name="Undercut", update=gear_definitions.update_gear, default=False)
    shift: FloatProperty(name="Shift", default=0.0, min=-1, max=1, update=gear_definitions.update_gear)
    complexity: IntProperty(name="Complexity", default=10, update=gear_definitions.update_gear, min=1, soft_max=25)
    head: FloatProperty(name="Head", default=0, min=-1.5, update=gear_definitions.update_gear)
    clearance: FloatProperty(name="Clearance", default=0.05, update=gear_definitions.update_gear)
    reverse_pitch: BoolProperty(name="Reverse Pitch", update=gear_definitions.update_gear, default=False)

    # Bore
    bore_enabled: BoolProperty(name="Bore Toggle", default=False, update=gear_definitions.update_gear)
    bore: EnumProperty(name="Bore", update=gear_definitions.update_gear, items=bores.enum, default=0)
    bore_size: FloatProperty(name="Bore Size", update=gear_definitions.update_gear, subtype="DISTANCE", precision=consts.UI_FLOAT_PRECISION, default=1, min=0.000000000001,)
    bore_param_a: FloatProperty(name="Bore Param A", subtype="DISTANCE", update=gear_definitions.update_gear, precision=consts.UI_FLOAT_PRECISION, default=1, min=0,)
    bore_param_b: FloatProperty(name="Bore Param B", subtype="DISTANCE", update=gear_definitions.update_gear, precision=consts.UI_FLOAT_PRECISION, default=1, min=0,)
    bore_resolution: IntProperty(name="Bore Resolution", update=gear_definitions.update_gear, default=8, min=3)
    bore_subtype: BoolProperty(name="Bore Subtype", update=gear_definitions.update_gear, subtype="DISTANCE", default=False)
    preview_bore: BoolProperty(name="Preview Bore", update=gear_definitions.update_gear, default=False)

    # Tooth Modification
    tooth_mod_enabled: BoolProperty(name="Tooth Toggle", default=False, update=gear_definitions.update_gear)
    tooth_taper: FloatProperty(name="Tooth Taper", subtype="DISTANCE", update=gear_definitions.update_gear, precision=consts.UI_FLOAT_PRECISION, default=0, min=0,)
    tooth_taper_offset: FloatProperty(name="Tooth Taper Offset", subtype="DISTANCE", update=gear_definitions.update_gear, precision=consts.UI_FLOAT_PRECISION, default=0,)
    tip_relief: FloatProperty(name="Tip Relief", subtype="DISTANCE", update=gear_definitions.update_gear, precision=consts.UI_FLOAT_PRECISION, default=0, min=0,)
    tip_relief_resolution: IntProperty(name="Tip Relief Resolution", update=gear_definitions.update_gear, default=2, min=1, soft_max=3)
    root_relief: FloatProperty(name="Root Relief", subtype="DISTANCE", update=gear_definitions.update_gear, precision=consts.UI_FLOAT_PRECISION, default=0, min=0,)
    root_relief_resolution: IntProperty(name="Root Relief Resolution", update=gear_definitions.update_gear, default=2, min=1, soft_max=3)
    worm_cut: FloatProperty(name="Worm Cut", subtype="DISTANCE", update=gear_definitions.update_gear, precision=consts.UI_FLOAT_PRECISION, default=0, min=0,)
    worm_cut_scale: FloatProperty(name="Worm Cut Scale", subtype="FACTOR", update=gear_definitions.update_gear, precision=consts.UI_FLOAT_PRECISION, default=1, min=0.01, soft_max=2,)
    flank_modification: EnumProperty(name="Flank Modification", update=gear_definitions.update_gear, items=consts.FLANK_MODS_ENUM, default="NONE",)
    flank_mod_param_a: FloatProperty(name="Flank Mod A", subtype="DISTANCE", update=gear_definitions.update_gear, precision=consts.UI_FLOAT_PRECISION, default=1, min=0,)
    flank_mod_param_b: FloatProperty(name="Flank Mod B", subtype="DISTANCE", update=gear_definitions.update_gear, precision=consts.UI_FLOAT_PRECISION, default=1, min=0,)

    # Set Screw Hole
    set_screw_enabled: BoolProperty(name="Set Screw Toggle", default=False, update=gear_definitions.update_gear)
    set_screw: IntProperty(name="Set Screw", update=gear_definitions.update_gear, default=0, min=0, max=len(consts.SET_SCREW_ARRANGEMENTS),)
    set_screw_offset: FloatProperty(name="Set Screw Offset", subtype="DISTANCE", update=gear_definitions.update_gear, default=1)
    set_screw_radius: FloatProperty(name="Set Screw Radius", subtype="DISTANCE", update=gear_definitions.update_gear, default=0.1, min=0,)
    set_screw_resolution: IntProperty(name="Set Screw Resolution", update=gear_definitions.update_gear, default=64, min=1, soft_max=16)
    set_screw_angle: FloatProperty(name="Set Screw Angle", subtype="ANGLE", update=gear_definitions.update_gear, default=0)
    preview_set_screw: BoolProperty(name="Preview Set Screw", update=gear_definitions.update_gear, default=False)

    # Extrusion
    extrusion_enabled: BoolProperty(name="Extrusion Toggle", default=False, update=gear_definitions.update_gear)
    extrusion_size: FloatProperty(name="Extrusion Size", subtype="DISTANCE", update=gear_definitions.update_gear, default=0, min=0)
    extrusion_radius: FloatProperty(name="Extrusion Radius", subtype="DISTANCE", update=gear_definitions.update_gear, default=0.5, min=0,)
    extrusion_top: BoolProperty(name="Extrusion Top", update=gear_definitions.update_gear, default=False)
    extrusion_bottom: BoolProperty(name="Extrusion Bottom", update=gear_definitions.update_gear, default=True)
    extrusion_resolution: IntProperty(name="Extrusion Resolution", update=gear_definitions.update_gear, default=64, min=1, soft_max=10)

    # General
    scaler: FloatProperty(name="Scaler", update=gear_definitions.update_gear, default=1, min=0)
    bool_solver: EnumProperty(name="Bool Solver", items=consts.BOOLEAN_TYPE_ENUM, update=gear_definitions.update_gear)
    do_crease: BoolProperty(name="Do Crease", update=gear_definitions.update_gear, default=True)
    crease: FloatProperty(name="Scaler", update=gear_definitions.update_gear, subtype="ANGLE", default=radians(35), min=0)
    add_weld: BoolProperty(name="Do Weld", update=gear_definitions.update_gear, default=True)
    weld_dist: FloatProperty(name="Weld", update=gear_definitions.update_gear, subtype="DISTANCE", min=0)
    unwrap: BoolProperty(name="UV Unwrap", update=uv_prop_callback, default=False)
    uv_seam_angle: FloatProperty(name="Seam Angle", subtype="ANGLE", update=uv_prop_callback, default=radians(35))
    # smooth_angle: FloatProperty(name="Smooth Shade", update=gear_definitions.update_shade_smooth_callback, subtype="ANGLE", min=0, max=radians(180), default=radians(30))

    # Rigging
    rigging_enabled: BoolProperty(name="Rigging Enabled", default=False, update=update_rigging)
    driver: PointerProperty(name="Driver", type=Object, update=update_rigging)
    rig_tree: PointerProperty(type=GeometryNodeTree)
    root_driver: PointerProperty(name="Root Driver", type=Object)
    rig_host: PointerProperty(name="Rigging Host", type=Object)
    rigging_node: StringProperty()  # Name of rigging node
    rigging_mode: EnumProperty(name="Mode", items=ui_rig_modes_enum, update=update_rigging)
    is_rigging_host: BoolProperty(default=False)
    # node_test: PointerProperty(name="Rigging Host", type=GeometryNodeGroup)
    # driver_offset: FloatProperty(name="Driver Offset", default=0)

    intersect_samples: IntProperty(name="Intersect Samples", default=30, min=10)  # Samples for automatic intersection correction operator
    intersect_solve_recursive: BoolProperty(default=False)
    intersect_solve_from_root: BoolProperty(default=False)
    # reverse_parallel_solve: BoolProperty(name="Revere Parallel Solve", default=False)  # TODO: Can I delete this? I don't know its purpose
    refresh_rig: BoolProperty(default=False, update=update_rigging)  # Set True to trigger a rigging update



class PrecisionGearsPreferences(AddonPreferences):
    bl_idname = __package__
    debug_level: EnumProperty(items=consts.LOG_LEVELS, default="WARNING")
    test_prop: StringProperty()

    def draw(self, ctx: Context):
        layout = self.layout
        layout.prop(self, "debug_level")


classes = (
    PrecisionGearsProps,
    PrecisionGearsPreferences
    # SceneProps
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    from bpy.utils import previews  # NOTE: Shouldn't be necesssary, seems a weird namespace issue with api?

    setattr(bpy.types.Object, consts.GEAR_PROPS_ALIAS, PointerProperty(type=PrecisionGearsProps))
    setattr(bpy.types.Scene, consts.PRESET_UPDATE_PROP_ALIAS, BoolProperty(default=False))

    preview_collection = previews.new()
    preview_collection.names = []
    preview_collection.filter = ""
    preview_collection.force_update = False
    preview_collection.active_filter = None
    preview_collection.active_directory = None
    preview_collection.n_known_thumbs = 0
    preview_collection.previews = ()
    preview_collections[consts.THUMB_COLLECTION_ALIAS] = preview_collection


def unregister():
    # Remove preview collections
    for collection in preview_collections.values():
        bpy.utils.previews.remove(collection)
    preview_collections.clear()

    delattr(bpy.types.Object, consts.GEAR_PROPS_ALIAS)
    delattr(bpy.types.Scene, consts.PRESET_UPDATE_PROP_ALIAS)

    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
