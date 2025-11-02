import bpy
from bpy.types import Context, Object
from . import consts


def is_lite_gear(obj: Object):
    def _tests():
        yield hasattr(obj, "lite_gear_props")
        yield obj.lite_gear_props.is_gear
    return all(_tests())

def is_active_lite_version_gear(context: Context):
    def _tests():
        yield context.active_object is not None
        yield hasattr(context.active_object, "lite_gear_props")
        yield context.active_object.lite_gear_props.is_gear
    return all(_tests())


def is_active_a_gear(context: Context):
    def _tests():
        yield context.active_object is not None
        gear_props = getattr(context.active_object, consts.GEAR_PROPS_ALIAS)
        yield gear_props.is_gear
        yield context.mode == "OBJECT"
    return all(_tests())
    
