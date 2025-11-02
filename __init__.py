import bpy 

from . import ui
from . import props
from . import operators
from . import async_loop


bl_info = {
    "name": "Precision Gears",
    "description": "Precision Gears",
    "author": "Missing Field <themissingfield.com>",
    "version": (0, 1, 93),
    "blender": (3, 00, 0),
    "location": "View3D",
    "category": "Object",
}

registration_queue = (props, operators, ui)
# from . import presets

def register():
    try:
        async_loop.setup_asyncio_executor()
    except Exception as e:
        print(f"Warning: Failed to setup asyncio executor: {e}")
        print("Addon will continue but async features may not work")

    try:
        bpy.utils.register_class(async_loop.AsyncLoopModalOperator)
    except RuntimeError as e:
        print(f"Warning: {e}")
    except ValueError as e:
        print('Value error registering Async Operator, likely already registered')

    for item in registration_queue:
        item.register()


def unregister():
    try:
        bpy.utils.unregister_class(async_loop.AsyncLoopModalOperator)
    except RuntimeError as e:
        print(e)
    except ValueError as e:
        print('Value error unregistering Async Operator, likely already registered')
        pass

    for item in registration_queue:
        item.unregister()


if __name__ == "__main__":
    register()
