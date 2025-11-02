from pathlib import Path
import bpy


_BPY_4_1_COMPAT_REQUIRED = bpy.app.version[0] >= 4 and bpy.app.version[1] > 0
_SMOOTH_GROUP_FILEPATH = Path(__file__).parent / "resources/smooth_by_angle_gn_group.blend"
_SMOOTH_GROUP_NAME = "Smooth by Angle"


def get_node_group() -> bpy.types.GeometryNodeGroup:
    if not bpy.data.node_groups.get(_SMOOTH_GROUP_NAME):
        with bpy.data.libraries.load(str(_SMOOTH_GROUP_FILEPATH), link=False) as (data_from, data_to):
            data_to.node_groups = [_SMOOTH_GROUP_NAME]
    return bpy.data.node_groups.get(_SMOOTH_GROUP_NAME)


def ensure_autosmooth_on_object(ob: bpy.types.Object) -> bpy.types.Modifier:
    if not _BPY_4_1_COMPAT_REQUIRED:
        ob.data.use_auto_smooth = True
    else:
        smooth_node_group = get_node_group()
        for modifier in ob.modifiers:
            if not modifier.type == "NODES":
                continue
            elif modifier.node_group != smooth_node_group:
                continue
            elif modifier.node_group is smooth_node_group:
                break
        else:
            smooth_mod = ob.modifiers.new("Smooth by Angle", type="NODES")
            smooth_mod.node_group = smooth_node_group
