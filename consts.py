from pathlib import Path
from math import radians

import bpy

def __enum_from_iter(iterable):
    enum = []
    for i, item in enumerate(iterable):
        enum.append((item, item.title(), item.title(), i))
    return enum

LOG_LEVELS = [
    ("CRITICAL", "CRITICAL", "CRITICAL", 0),
    ("ERROR", "ERROR", "ERROR", 1),
    ("WARNING", "WARNING", "WARNING", 2),
    ("INFO", "INFO", "INFO", 3),
    ("DEBUG", "DEBUG", "DEBUG", 4),
]

DEBUG_PRINT_THUMB_RENDER_LOG = True
SCRIPT_DIR = Path(__file__).parent
BLENDER = bpy.app.binary_path

### Property Aliases ###
GEAR_PROPS_ALIAS = "gear"
GEAR_ATTRIBS_ALIAS = "gear_attribs"
GEAR_SCENE_PROPS = "gear_scene_props"

BPY_4_1_COMPAT_REQUIRED = bpy.app.version[0] >= 4 and bpy.app.version[1] > 0

### Enums ###
EMPTY_ENUM = [("NONE", "None", "None", 0)]
GEAR_TYPES = ("RACK", "SPUR", "INTERNAL", "BEVEL", "WORM", "GT2GEAR",)
GEAR_TYPES_ENUM = __enum_from_iter(GEAR_TYPES)
BOOLEAN_TYPES = ("FAST", "EXACT")
BOOLEAN_TYPE_ENUM = __enum_from_iter(BOOLEAN_TYPES)
RIGGING_MODES = ("MESHING", "COMPOUND")
RIGGING_MODES_ENUM = __enum_from_iter(RIGGING_MODES)
FLANK_MODIFICATIONS = ("NONE", "CROWNING", "END_RELIEF")
FLANK_MODS_ENUM = __enum_from_iter(FLANK_MODIFICATIONS)

# Drivers compatible with a given gear type
VALID_RIG_DRIVERS = {
    "SPUR": {
        "ROTATION": {"SPUR", "INTERNAL", "WORM", "RACK"},
        "COMPOUND": {"SPUR", "INTERNAL", "BEVEL", "WORM", "GT2GEAR"},
    },
    "GT2GEAR": {
        "ROTATION": {"GT2GEAR",},
        "COMPOUND": {"SPUR", "INTERNAL", "BEVEL", "WORM"},
    },
    "INTERNAL": {
        "ROTATION": {"SPUR", "INTERNAL"},
        "COMPOUND": {"SPUR", "INTERNAL", "BEVEL", "WORM", "GT2GEAR"},
    },
    "RACK": {
        "LINEAR": {"SPUR",},
    },
    "BEVEL": {
        "BEVEL": {"BEVEL",},
        "ROTATION": {"BEVEL",},
        "COMPOUND": {"SPUR", "INTERNAL", "BEVEL", "WORM", "GT2GEAR"},
    },
    "WORM": {
        "ROTATION": {"SPUR",},
        "COMPOUND": {"SPUR", "INTERNAL", "BEVEL", "WORM", "GT2GEAR"},
    },
}

RIG_NODE_INTERSECTION_PROP = {
    "SPUR": "Rot Offset",
    "GT2GEAR": "Rot Offset",
    "INTERNAL": "Rot Offset",
    "RACK": "Rot Offset",
    "WORM": "Rot Offset",
    "BEVEL": "Rot Offset",
    "RACK": "Offset",
}

### UI ###
UI_FLOAT_PRECISION = 4

### Thumb rendering ###
THUMB_GEN_IP = "127.0.0.1"
THUMB_GEN_PORT = "8888"
RESOURCES_DIR = SCRIPT_DIR / "resources"
PRESET_UPDATE_PROP_ALIAS = "gear_presets_update_required"
PRESETS_DIR = SCRIPT_DIR / "presets"
PRESET_THUMB_SCRIPT = SCRIPT_DIR / "generate_thumb.py"
PRESET_THUMB_TEMPLATE = RESOURCES_DIR / "thumb_render_template.blend"
PRESET_THUMB_RESOLUTION = 256
# PRESET_THUMB_DIR = USER_PRESETS_DIR / "thumbnails"
PRESET_THUMB_SUFFIX = ".png"
THUMB_COLLECTION_ALIAS = "gear_thumbnails"
# ACTIVE_PRESET_DIR_ALIAS = "active_fastener_preset_dir"

### Presets ###
# Files listed here cannot be deleted from the UI
# If the files are deleted manually, they can be created from the ui but not overwriten once created
PROTECTED_PRESETS = (
    "BEVEL/Default.csv",
    "INTERNAL/Default.csv",
    "RACK/Default.csv",
    "SPUR/Default.csv",
    "WORM/Default.csv",
    "GT2/Default.csv",
)

# These are the set screw patterns
SET_SCREW_ARRANGEMENTS = (
    (0,),
    (0, radians(90)),
    (0, radians(180)),
    (0, radians(90), radians(180), radians(270)),
)

### Rigging ###
RIGGING_NODE_GROUPS_PATH = RESOURCES_DIR / "rigging_gn_groups.blend"
RIGGING_NODE_GROUPS = {
    "ROTATION": "ROTATION",
    "LINEAR": "LINEAR",
    "COMPOUND": "COMPOUND",
    "BEVEL": "BEVEL",
    "GEAR_STATE_CONTROL": "GEAR_STATE_CONTROL",
    "PG_LOCATOR": "PG_LOCATOR",
}
ROOT_DRIVER_NODE_NAME = "ROOT_DRIVER"
ROOT_DRIVER_STATE_NODE_NAME = "GEAR_STATE_CONTROL"


LITE_VERSION_ATTRIBS = (
    # "gear_type",
    "teeth",
    "module",
    "height",
    "pressure_angle",
    "shift",
    "head",
    "backlash",
    "clearance",
    "complexity",
    "bore",
    "bore_param_a",
    "bore_param_b",
    "bore_size",
    "bore_subtype",
    "bore_resolution",
    "scaler",
    "bool_solver",
    # "undercut"
)