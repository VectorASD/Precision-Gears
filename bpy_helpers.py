import bpy
from bpy.types import UILayout, Context, PropertyGroup
from bpy.props import BoolProperty


def pop_message(message="", title="Message Box", icon="INFO"):
    def draw(self: UILayout, context: Context):
        self.layout.label(text=message)

    if not bpy.app.background:
        bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)
    else:
        print(message)


def pop_error(message=""):
    """Wrapper for pop message with error icon and title"""
    pop_message(message, title="ERROR", icon="ERROR")


class PropsUpdateDisabled:
    """
    Context managed for toggling property group update function off
    Give property group a bool property that it can refer to that will
    be recognized by the update function as a bypass for calls to it's True
    """

    def __init__(self, props: PropertyGroup, update_prop: BoolProperty):
        self.props = props
        self.update_prop = update_prop

    def __enter__(self):
        # self.props.editing = True
        setattr(self.props, self.update_prop, False)
        return self.props

    def __exit__(self, *args):
        setattr(self.props, self.update_prop, True)
        # self.props.editing = False
