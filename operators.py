from collections import deque
from functools import partial
from pathlib import Path
from typing import Iterator
import functools
from csv import DictWriter
import asyncio
import os
import subprocess
import platform 

import bmesh
import bpy
from mathutils import Matrix
from bpy.path import clean_name
from bpy.types import Event, Operator, Context, Object, GeometryNodeTree
from bpy.props import EnumProperty, StringProperty, IntProperty, BoolProperty

from . import (
    async_loop,
    mesh_generation,
    presets,
    gear_definitions,
    consts,
    polls,
    preset_prop_groups,
    props,
    rigging,
)
from .consts import BPY_4_1_COMPAT_REQUIRED
# from . import rigging

# class OBJECT_OT_test_operator(Operator):
#     bl_idname = "object.precision_gears_test_operator"
#     bl_label = "TESTING OPERATOR"
#     bl_description = "NOTHING"
#     bl_options = {"REGISTER", "UNDO"}

#     def execute(self, context: Context):
#         try:
#             # print("No test code")
#             overlap_test(context, *context.selected_objects[:2])
#         except Exception as e:
#             print(e)
#         return {"FINISHED"}


class RENDER_OT_generate_gear_thumbnails(Operator):
    """ Async generate thumbnails in background subprocess """
    bl_idname = "render.render_gear_thumbnails"
    bl_label = "Generate Precision Gear Thumbnails"
    bl_options = {"UNDO"}

    async def generate_thumbs(self):
        await presets.generate_thumbnails()

    @staticmethod
    def _completion_callback(props, _):
        # show_ui_message_popup("Thumbnail Render Complete")
        # properties.trigger_preset_refresh(props)
        props.thumb_rendering = False

    def execute(self, context: Context):
        self.report({"INFO"}, "Generating thumbnails")

        props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
        props.thumb_rendering = True

        # Ensure event loop exists before scheduling async task
        # (loop parameter deprecated in Python 3.10+)
        async_loop._get_event_loop()
        async_task = asyncio.ensure_future(self.generate_thumbs())
        callback = partial(self._completion_callback, props)
        async_task.add_done_callback(callback)
        async_loop.ensure_async_loop()
        return {"FINISHED"}


class OBJECT_OT_solve_gear_intersection(Operator):
    bl_idname = "object.solver_gear_intersection"
    bl_label = "Solve Gear Intersection"
    bl_description = "Solve rigged gear intersections"
    bl_options = {"REGISTER", "UNDO"}

    recursive: BoolProperty(default=True)
    from_root: BoolProperty(default=False)
    samples: IntProperty(name="Intersect Samples", default=30, min=10)
    # mode: EnumProperty(name="Parallel", items=consts.RIGGING_MODES_ENUM)
    # reverse_parallel: BoolProperty(name="Flip Parallel", default=False)

    @classmethod
    def poll(cls, context: Context) -> bool:
        def _conditions():
            yield context.active_object is not None
            yield context.active_object.type == "MESH"
            gear_props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
            yield gear_props.driver is not None
        return all(_conditions())

    def _report_failure(self) -> None:
        warning = "".join((
            "Imperfect Solve. ",
            "Suggest Match Driver ",
            "and adjusting backlash and/or clearance",
        ))
        self.report(type={"WARNING"}, message=warning)

    def execute(self, ctx: Context):
        rigging.correct_intersection(ctx, ctx.active_object, self.recursive, self.from_root, self.samples)

        # if not success:
            # self._report_failure()
        return {"FINISHED"}


class OBJECT_OT_make_gear_compatible(Operator):
    bl_idname = "object.make_gear_compatible"
    bl_label = "Make Gear Compatible"
    bl_description = "Make gear compatible with its driver"
    bl_options = {"REGISTER", "UNDO"}

    recursive: BoolProperty(default=False)

    @classmethod
    def poll(cls, context: Context):
        def _conditions():
            yield context.active_object is not None
            yield context.active_object.type == "MESH"
            gear_props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
            yield gear_props.driver is not None
        return all(_conditions())

    @property
    def _all_driven_gears(self) -> Iterator:
        """ Return all driven gears """
        def _has_driver(obj: Object) -> bool:
            props = getattr(obj, consts.GEAR_PROPS_ALIAS)
            return props.driver is not None
        return filter(_has_driver, bpy.data.objects)

    def _make_compatible(self, context, gear):
        rigging.make_driven_compatible(gear)

    def execute(self, context: Context):
        first_gear = context.active_object

        if not self.recursive:
            self._make_compatible(context, first_gear)
        else:
            process_queue = deque((first_gear,))
            unseen_driven_gears = set(self._all_driven_gears)

            while process_queue:
                gear = process_queue.pop()
                self._make_compatible(context, gear)

                driven_gears = set((
                    driven for driven in unseen_driven_gears
                    if getattr(driven, consts.GEAR_PROPS_ALIAS).driver is gear
                ))
                unseen_driven_gears = unseen_driven_gears.difference(driven_gears)
                process_queue.extend(driven_gears)

        return {"FINISHED"}


class OBJECT_OT_apply_gear_preset(Operator):
    bl_idname = "object.apply_gear_preset"
    bl_label = "Apply gear preset"
    bl_description = "Apply preset to active gear"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context):
        def _conditions():
            yield context.active_object is not None
            yield context.active_object.type == "MESH"
            gear_props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
            yield gear_props.is_gear
            # yield gear_props.driver is not None
        return all(_conditions())
        # props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
        # return props.is_gear

    def execute(self, context: Context):
        props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
        props.apply_preset(context)
        return {"FINISHED"}


class OBJECT_OT_add_new_gear(Operator):
    bl_idname = "object.add_new_gear"
    bl_label = "Add Gear"
    bl_description = "Add a New Gear"
    bl_options = {"REGISTER", "UNDO"}

    gear_type: EnumProperty(items=consts.GEAR_TYPES_ENUM)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Edit Gear Properties in Object Data Panel")

    def execute(self, context: Context):
        xform = context.scene.cursor.matrix

        # Create placeholder object
        gear_mesh = mesh_generation.new_grid_mesh(
            "placehodler", transform=xform)

        if not BPY_4_1_COMPAT_REQUIRED:
            gear_mesh.use_auto_smooth = True
        gear_name = self.gear_type[0] + self.gear_type.lower()[1:]
        gear_object = bpy.data.objects.new(name=f"{gear_name} Gear", object_data=gear_mesh)
        gear_object.matrix_world = context.scene.cursor.matrix
        context.collection.objects.link(gear_object)

        # Ensuring editing flag set to prevent full prop update call
        gear_props = getattr(gear_object, consts.GEAR_PROPS_ALIAS)
        gear_props.editing = True
        gear_props.is_gear = True
        gear_props.gear_type = self.gear_type
        gear_props.editing = False
        builder = gear_definitions.get_gear_definition(gear_props.gear_type)(gear_props)
        builder.set_prop_defaults()
        context.view_layer.objects.active = gear_object
        gear_definitions.update_gear(gear_props, context)
        return {"FINISHED"}


class OBJECT_OT_upgrade_lite_gear(Operator):
    bl_idname = "object.upgrade_lite_gear"
    bl_label = "Upgrade Lite Gear"
    bl_description = "Upgrade Lite Gear to full version gear"
    bl_options = {"REGISTER", "UNDO"}

    batch_update: BoolProperty(default=False)
    only_selected: BoolProperty(default=False)

    @classmethod
    def poll(cls, context: Context):
        return polls.is_active_lite_version_gear(context)

    def _upgrade_gear(self, context, gear):
        input_props = {}
        for attrib in consts.LITE_VERSION_ATTRIBS:
            input_props[attrib] = getattr(gear.lite_gear_props, attrib)
        
        gear.lite_gear_props.is_gear = False

        gear_props = getattr(gear, consts.GEAR_PROPS_ALIAS)
        gear_props.editing = True
        gear_props.is_gear = True
        gear_props.gear_type = "SPUR"

        for attrib, value in input_props.items():
            if attrib == "bore":
                if value != "NONE":
                    gear_props.bore_enabled = True
                    gear_props.bore = value
            else:
                try:
                    setattr(gear_props, attrib, value)
                except Exception as e:
                    pass

        gear_props.editing = False
        gear_definitions.update_gear(gear_props, context)

    def execute(self, context: Context):
        if not self.batch_update:
            target_gear = bpy.context.active_object
            self._upgrade_gear(context, target_gear)
        else:
            for target in filter(polls.is_lite_gear, bpy.data.objects):
                if self.only_selected and not target.select_get():
                    continue
                print("Upgrading lite gear", target)
                self._upgrade_gear(context, target)
        return {"FINISHED"}


class OBJECT_OT_save_gear_preset(Operator):
    bl_idname = "object.save_gear_preset"
    bl_label = "Save Gear Peset"
    bl_description = "Save a gear preset"
    bl_options = {"REGISTER"}

    preset_name: StringProperty(name="Preset Name", default="")
    gear_type: EnumProperty(items=consts.GEAR_TYPES_ENUM, options={"HIDDEN"})

    def invoke(self, context: Context, event: Event):
        if not polls.is_active_a_gear(context):
            self.report({"ERROR"}, "Active object is not a gear")
            return {"CANCELLED"}
        gear = context.active_object
        gear_props = getattr(gear, consts.GEAR_PROPS_ALIAS)
        self.gear_type = gear_props.gear_type
        return context.window_manager.invoke_props_dialog(self)

    @property
    def preset_path(self) -> Path:
        return consts.PRESETS_DIR / self.gear_type / (self.preset_name + ".csv")

    def ensure_output_directory(self):
        preset_directory = consts.PRESETS_DIR / self.gear_type
        preset_directory.mkdir(parents=True, exist_ok=True)

    def execute(self, context: Context):
        if self.preset_name == "":
            self.report({"ERROR"}, "No valid name provided")
            return {"CANCELLED"}

        if self.preset_path.exists():
            self.report({"ERROR"}, f"{self.preset_path} already exists")
            return {"CANCELLED"}

        self.ensure_output_directory()
        gear = context.active_object
        gear_props = getattr(gear, consts.GEAR_PROPS_ALIAS)
        builder = gear_definitions.get_gear_definition(gear_props.gear_type)
        fieldnames = [key for key in gear_props.bl_rna.properties.keys() if key not in preset_prop_groups.PRESET_FIELD_BLOCKLIST]  # Get used props
        fieldnames.insert(0, "preset_name")
        fieldnames.insert(1, "thumb")

        try: 
            with open(self.preset_path, 'w') as preset_file:
                writer = DictWriter(preset_file, fieldnames=fieldnames)
                writer.writeheader()
                preset = {}
                preset["preset_name"] = self.preset_name
                preset["thumb"] = clean_name(self.preset_name)
                for key in fieldnames:
                    gear_prop = builder.get_prop_map().get(key, None)
                    if gear_prop is not None:
                        preset[key] = getattr(gear_props, key)
                writer.writerow(preset)

        except Exception as e:
            self.preset_path.unlink()
            self.report({"ERROR"}, f"{e} occured while writing {self.preset_path}")
            print(e)
            return {"CANCELLED"}

        bpy.ops.render.render_gear_thumbnails()
        return {"FINISHED"}


class OBJECT_OT_delete_gear_preset(Operator):
    bl_idname = "object.delete_gear_preset"
    bl_label = "Delete Gear Preset"
    bl_description = "Delete a gear preset"
    bl_options = {"REGISTER"}

    preset_name: StringProperty(name="Preset Name", default="")
    gear_type: EnumProperty(items=consts.GEAR_TYPES_ENUM, options={"HIDDEN"})

    @property
    def preset_path(self) -> Path:
        return consts.PRESETS_DIR / self.gear_type / (self.preset_name + ".csv")

    def is_protected_preset(self):
        protected_suffixes = consts.PROTECTED_PRESETS
        for suffix in protected_suffixes:
            if str(self.preset_path).endswith(suffix):
                return True
        return False

    def execute(self, context: Context):
        if not polls.is_active_a_gear(context):
            self.report({"ERROR"}, "Active object is not a gear")
            return {"CANCELLED"}

        gear = context.active_object
        gear_props = getattr(gear, consts.GEAR_PROPS_ALIAS)

        if gear_props.preset == "NONE":
            self.report({"ERROR"}, "No active preset")
            return {"CANCELLED"}

        self.preset_name = gear_props.preset
        self.gear_type = gear_props.gear_type

        if self.is_protected_preset():
            self.report({"ERROR"}, "Preset is protected")
            return {"CANCELLED"}
        
        self.preset_path.unlink()
        self.report({"INFO"}, f"Removed {self.preset_name}")
        return {"FINISHED"}


class OBJECT_OT_apply_gear_preset_filter(Operator):
    """ Load threads preset and apply """

    bl_idname = "object.apply_gear_preset_filter"
    bl_label = "Apply Fastener Preset Filter"
    bl_property = "preset"
    preset: bpy.props.EnumProperty(items=props.populate_thumb_enum)

    def invoke(self, context: Context, event: Event):
        wm = context.window_manager
        wm.invoke_search_popup(self)
        return {"FINISHED"}

    def execute(self, context: Context):
        props = getattr(context.active_object, consts.GEAR_ATTRIBS_ALIAS)
        props.preset_filter = self.preset
        props.preset_thumbnail = self.preset
        props.apply_preset_filter = True
        return {"FINISHED"}


class OBJECT_OT_refresh_gear_rig(Operator):
    """ Load threads preset and apply """

    bl_idname = "object.refresh_gear_rig"
    bl_label = "Refresh Gear Rig"

    def execute(self, context: Context):
        props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
        props.refresh_rig = True
        return {"FINISHED"}


class OBJECT_OT_cursor_to_rigged_gear(Operator):
    """ Load threads preset and apply """

    bl_idname = "object.pg_cursor_to_rigged"
    bl_label = "Cursor to Rigged"

    # gear: StringProperty()
    rig_host_name: StringProperty()
    tree_name: StringProperty()
    rig_node_name: StringProperty()

    def execute(self, ctx: Context):
        try:
            tree: GeometryNodeTree = bpy.data.node_groups.get(self.tree_name)
            rig_node = tree.nodes.get(self.rig_node_name)
            rig_host: Object = ctx.view_layer.objects.get(self.rig_host_name)
        except Exception as e:
            print(e)
            return {"CANCELLED"}

        # Get tree output node
        output_node = tree.nodes.get("Group Output")

        # Store initial output link connections
        init_link = output_node.inputs["Geometry"].links[0]
        init_from_node = init_link.from_node
        init_from_socket = init_link.from_socket
        tree.links.remove(init_link)

        # Setup Pointer node
        rigging.ensure_rig_node_groups_linked()

        locator_node = tree.nodes.new("GeometryNodeGroup")
        locator_node.node_tree = bpy.data.node_groups["PG_LOCATOR"]
        for data in ("Origin", "Y", "Z"):
            tree.links.new(rig_node.outputs[data], locator_node.inputs[data])
        tree.links.new(locator_node.outputs[0], output_node.inputs[0])

        # Eval tree and get location in scene space
        dg = ctx.evaluated_depsgraph_get()
        evaled_rig = rig_host.evaluated_get(dg)
        mesh = evaled_rig.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(mesh)

        # Calculate new cursor transform
        n_verts = len(bm.verts)
        locs = [v.co for v in bm.verts]
        sum_locs = functools.reduce(lambda a,b: a+b, locs)
        origin = sum_locs / float(n_verts)

        x = (locs[1] - origin).normalized()
        y = (locs[0] - origin).normalized()
        z = x.cross(y)
        # print(x, y, z)
        m = Matrix([
            (x.x, x.y, x.z, origin.x),
            (y.x, y.y, y.z, origin.y),
            (z.x, z.y, z.z, origin.z),
            (0.0, 0.0, 0.0, 1.0),
        ])
        
        m = rig_host.matrix_world @ m

        ctx.scene.cursor.matrix = m
        bm.free()

        # Remove locator node
        tree.nodes.remove(locator_node)

        # Restore initial output link connections
        tree.links.new(init_from_socket, output_node.inputs[0])
        return {"FINISHED"}


class WM_OT_gears_open_sys_folder(bpy.types.Operator):
    """ Save fastener preset threads preset and apply """
    bl_idname = "wm.gears_open_sys_folder"
    bl_label = "Open System Folder"

    folder: StringProperty(default="")

    def execute(self, context: Context):
        folder = os.path.realpath(self.folder)
        if not os.path.isdir(folder):
            self.report({"WARNING"}, f"{folder} not a directory")
            return {"CANCELLED"}
        system = platform.system()
        if system == "Darwin":
            subprocess.call(("open", folder))
        elif system == "Windows":
            os.startfile(folder)
        else:
            subprocess.call(("xdg-open", folder))
        return {"FINISHED"}



OPERATORS = (
    # OBJECT_OT_test_operator,
    OBJECT_OT_upgrade_lite_gear,
    OBJECT_OT_add_new_gear,
    OBJECT_OT_refresh_gear_rig,
    OBJECT_OT_apply_gear_preset_filter,
    OBJECT_OT_apply_gear_preset,
    OBJECT_OT_solve_gear_intersection,
    OBJECT_OT_make_gear_compatible,
    OBJECT_OT_save_gear_preset,
    OBJECT_OT_delete_gear_preset,
    OBJECT_OT_cursor_to_rigged_gear,
    RENDER_OT_generate_gear_thumbnails,
    WM_OT_gears_open_sys_folder,
)


def register():
    for op in OPERATORS:
        bpy.utils.register_class(op)


def unregister():
    for op in OPERATORS:
        bpy.utils.unregister_class(op)


if __name__ == "__main__":
    register()
