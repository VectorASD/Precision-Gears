import bpy
from bpy.types import (
    Context,
    UILayout,
    Menu,
    Panel,
)
from . import (
    gear_definitions,
    consts,
    operators,
    polls,
)


class GearObjectPanel:
    """ Root Object Panel"""
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    def get_builder(self, props):
        return gear_definitions.get_gear_definition(props.gear_type)

    @classmethod
    def get_props(cls, context):
        gear = context.active_object
        return getattr(gear, consts.GEAR_PROPS_ALIAS)


class OBJECT_PT_UpgradeLiteGear(GearObjectPanel, Panel):
    """Object panel edit active gear"""
    bl_label = "Precision Gears"
    bl_idname = "OBJECT_PT_UpgradeLiteGear"

    @classmethod
    def poll(cls, context):
        return polls.is_active_lite_version_gear(context)

    def draw(self, context: Context):
        layout: UILayout = self.layout
        layout.label(text="Upgrade Lite Gear")

        col = layout.column(align=True)

        # Update only active
        col.operator("object.upgrade_lite_gear", text="Active").batch_update = False

        # Update all selected
        update_selected = col.operator("object.upgrade_lite_gear", text="Selected")
        update_selected.batch_update = True
        update_selected.only_selected = True

        # Update all in file
        batch_update = col.operator("object.upgrade_lite_gear", text="All")
        batch_update.batch_update = True
        batch_update.only_selected = False

        # props = self.get_props(context)
        # builder = self.get_builder(props)
        # builder.draw_header(self.layout, props)


class OBJECT_PT_EditGear(GearObjectPanel, Panel):
    """Object panel edit active gear"""
    bl_label = "Precision Gears"
    bl_idname = "OBJECT_PT_EditGear"

    @classmethod
    def poll(cls, context):
        return polls.is_active_a_gear(context)

    def draw(self, context: Context):
        props = self.get_props(context)
        builder = self.get_builder(props)
        builder.draw_header(self.layout, props)


# class OBJECT_PT_RigObject(GearObjectPanel, Panel):
#     """Object panel edit active gear"""
#     bl_label = "Precision Gears"
#     bl_idname = "OBJECT_PT_RigObject"

#     @classmethod
#     def poll(cls, context):
#         return not all(polls.is_active_a_gear(cls, context))

#     def draw(self, context: Context):
#         props = self.get_props(context)
#         layout: UILayout = self.layout
#         row = layout.row()
#         box = row.box()
#         col = box.column(align=True)
#         # col.prop(props, "is_rigged_object", text="Gear Rigging")
#         # props = self.get_props(context)
#         # builder = self.get_builder(props)
#         # builder.draw_header(self.layout, props)


class OBJECT_PT_EditGearCoreParams(GearObjectPanel, Panel):
    """Gear Core Parameters"""
    bl_label = "Parameters"
    bl_idname = "OBJECT_PT_EditGearCoreParameters"
    bl_parent_id = OBJECT_PT_EditGear.bl_idname

    def draw(self, context: Context):
        layout = self.layout
        props = self.get_props(context)
        builder = self.get_builder(props)
        builder.draw_prop_set(layout, builder.gear_props, props)


class OBJECT_PT_EditGearToothMods(GearObjectPanel, Panel):
    """Gear Core Parameters"""

    bl_label = "Tooth Modification"
    bl_idname = "OBJECT_PT_EditGearToothMods"
    bl_parent_id = OBJECT_PT_EditGear.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: Context):
        compatible_gears = ("SPUR", "RACK", "INTERNAL")
        gear_props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
        return gear_props.gear_type in compatible_gears

    def draw_header(self, context):
        props = self.get_props(context)
        self.layout.prop(props, "tooth_mod_enabled", text="")

    def draw(self, context: Context):
        layout = self.layout
        props = self.get_props(context)
        layout.enabled = props.tooth_mod_enabled
        builder = self.get_builder(props)
        builder.draw_tooth_modifier_props(layout, props)


class OBJECT_PT_EditGearBore(GearObjectPanel, Panel):
    """Gear Core Parameters"""
    bl_label = "Bore"
    bl_idname = "OBJECT_PT_EditGearBore"
    bl_parent_id = OBJECT_PT_EditGear.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: Context):
        compatible_gears = ("SPUR", "WORM", "BEVEL", "GT2GEAR")
        gear_props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
        return gear_props.gear_type in compatible_gears

    def draw_header(self, context):
        props = self.get_props(context)
        self.layout.prop(props, "bore_enabled", text="")

    def draw(self, context: Context):
        layout = self.layout
        props = self.get_props(context)
        layout.enabled = props.bore_enabled
        builder = self.get_builder(props)
        builder.draw_bore_props(layout, props)


class OBJECT_PT_EditGearExtrusion(GearObjectPanel, Panel):
    """Gear Core Parameters"""
    bl_label = "Extrusion"
    bl_idname = "OBJECT_PT_EditGearExtrusion"
    bl_parent_id = OBJECT_PT_EditGear.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: Context):
        compatible_gears = ("SPUR", "WORM", "BEVEL", "GT2GEAR")
        gear_props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
        return gear_props.gear_type in compatible_gears

    def draw_header(self, context):
        props = self.get_props(context)
        self.layout.prop(props, "extrusion_enabled", text="")

    def draw(self, context: Context):
        layout = self.layout
        props = self.get_props(context)
        layout.enabled = props.extrusion_enabled
        builder = self.get_builder(props)
        builder.draw_prop_set(layout, builder.extrusion_props, props)


class OBJECT_PT_EditGearSetScrew(GearObjectPanel, Panel):
    """Gear Core Parameters"""
    bl_label = "Set Screw"
    bl_idname = "OBJECT_PT_EditGearSetScrew"
    bl_parent_id = OBJECT_PT_EditGear.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: Context):
        compatible_gears = ("SPUR", "WORM", "BEVEL", "GT2GEAR")
        gear_props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
        return gear_props.gear_type in compatible_gears

    def draw_header(self, context):
        props = self.get_props(context)
        self.layout.prop(props, "set_screw_enabled", text="")

    def draw(self, context: Context):
        layout = self.layout
        props = self.get_props(context)
        layout.enabled = props.set_screw_enabled
        builder = self.get_builder(props)
        builder.draw_prop_set(layout, builder.set_screw_props, props)


class OBJECT_PT_EditGearGeneral(GearObjectPanel, Panel):
    """Gear General Parameters"""
    bl_label = "General"
    bl_idname = "OBJECT_PT_EditGearGeneral"
    bl_parent_id = OBJECT_PT_EditGear.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: Context):
        layout = self.layout
        props = self.get_props(context)
        builder = self.get_builder(props)
        builder.draw_prop_set(layout, builder.extra_props, props)


class OBJECT_PT_EditGearRigging(GearObjectPanel, Panel):
    """Gear General Parameters"""
    bl_label = "Rigging"
    bl_idname = "OBJECT_PT_EditGearRigging"
    bl_parent_id = OBJECT_PT_EditGear.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        props = self.get_props(context)
        self.layout.prop(props, "rigging_enabled", text="")

    def draw(self, context: Context):
        layout: UILayout = self.layout
        props = self.get_props(context)
        builder = self.get_builder(props)
        builder.draw_rigging_props(layout, props)


class OBJECT_PT_EditGearAttributes(GearObjectPanel, Panel):
    """Gear General Parameters"""
    bl_label = "Attributes"
    bl_idname = "OBJECT_PT_EditGearAttributes"
    bl_parent_id = OBJECT_PT_EditGear.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    # def draw_header(self, context):
    #     props = self.get_props(context)
    #     self.layout.prop(props, "rigging_enabled", text="")

    def draw(self, context: Context):
        layout: UILayout = self.layout
        attribs = context.active_object[consts.GEAR_ATTRIBS_ALIAS]

        box = layout.box()
        col = box.column(align=True)
        for attrib, value in attribs.items():
            if hasattr(value, "__len__"):
                value = tuple(value)
            else:
                value = f"{value:.4f}"
            attrib = attrib.title().replace("_", " ")
            col.label(text=f"{attrib}: {value}")


class VIEW3D_MT_AddGear(Menu):
    bl_idname = "VIEW3D_MT_AddGear"
    bl_label = "Precision Gear"

    def draw(self, layout):
        layout = self.layout
        layout.separator()

        for gear_type in consts.GEAR_TYPES_ENUM:
            text = f"{gear_type[1]} Gear"
            op = layout.operator(
                operators.OBJECT_OT_add_new_gear.bl_idname, text=text, icon="LIGHT_SUN",
            )
            op.gear_type = gear_type[0]


def add_gear_menu_items(self, context: Context):
    layout = self.layout
    layout.separator()
    layout.operator_context = "INVOKE_REGION_WIN"
    layout.menu(VIEW3D_MT_AddGear.bl_idname)


ui_classes = [
    VIEW3D_MT_AddGear,
]

for cls in GearObjectPanel.__subclasses__():
    ui_classes.append(cls)


def register():
    for cls in ui_classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_mesh_add.append(add_gear_menu_items)


def unregister():
    bpy.types.VIEW3D_MT_mesh_add.remove(add_gear_menu_items)
    for cls in ui_classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
