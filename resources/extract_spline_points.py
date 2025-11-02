from typing import Generator, Tuple

import bpy
from bpy.types import Spline
from mathutils import Vector


def spline_as_bezier_sections(bezier: Spline) -> Generator[Tuple[Vector], None, None]:
    """
    Convert spline to bezier sections for use with mathutils bezier interp functions
    (co1, knot1, knot2, co2)
    """
    for spline in bezier.splines:
        for p1, p2 in zip(spline.bezier_points, spline.bezier_points[1:]):
            yield p1.co, p1.handle_right, p2.handle_left, p2.co


if __name__ == "__main__":
    # rounded_sections = []
    xy_sections = []
    for section in spline_as_bezier_sections(bpy.context.active_object.data):
        xy_section = []
        xy_sections.append([v.xy for v in section])

    for section in xy_sections:
        print([(round(x, 3), round(y, 3)) for x, y in section[:]])


# profile_sections = (
#     [Vector((-0.0, 0.0)), Vector((0.16500000655651093, 0.0)), Vector((0.4154999852180481, 0.07580006122589111)), Vector((0.5321000218391418, 0.390500009059906))]
#     [Vector((0.5321000218391418, 0.390500009059906)), Vector((0.5964950323104858, 0.5643001794815063)), Vector((0.5825999975204468, 0.6458999514579773)), Vector((0.6176999807357788, 0.6919000148773193))]
#     [Vector((0.6176999807357788, 0.6919000148773193)), Vector((0.6435729265213013, 0.7258076071739197)), Vector((0.6884999871253967, 0.7549999952316284)), Vector((0.7418000102043152, 0.7549999952316284))]
# )
