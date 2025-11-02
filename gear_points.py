"""
Generator of different gear types. 

Types:
    rack
        rack(modul, length, height, width, pressure_angle=20, helix_angle=0)
        mountable_rack(modul, length, height, width, pressure_angle=20, helix_angle=0, fastners, profile, head)
        herringbone_rack(modul, length, height, width, pressure_angle = 20, helix_angle=45)
        mountable_herringbone_rack(modul, length, height, width, pressure_angle=20, helix_angle=45, fastners, profile, head)
    spur
        spur_gear(modul, tooth_number, width, bore, pressure_angle=20, helix_angle=0, optimized=true)
        herringbone_gear(modul, tooth_number, width, bore, pressure_angle=20, helix_angle=0, optimized=true)
        rack_and_pinion (modul, rack_length, gear_teeth, rack_height, gear_bore, width, pressure_angle=20, helix_angle=0, together_built=true, optimized=true)
    internal
        ring_gear(modul, tooth_number, width, rim_width, pressure_angle=20, helix_angle=0)
        planetary_gear(modul, sun_teeth, planet_teeth, number_planets, width, rim_width, bore, pressure_angle=20, helix_angle=0, together_built=true, optimized=true)
        herringbone_ring_gear(modul, tooth_number, width, rim_width, pressure_angle=20, helix_angle=0)
    bevel gear
        bevel_gear(modul, tooth_number,  partial_cone_angle, tooth_width, bore, pressure_angle=20, helix_angle=0)
        bevel_herringbone_gear(modul, tooth_number, partial_cone_angle, tooth_width, bore, pressure_angle=20, helix_angle=0)
        bevel_gear_pair(modul, gear_teeth, pinion_teeth, axis_angle=90, tooth_width, bore, pressure_angle = 20, helix_angle=0, together_built=true)
        bevel_herringbone_gear_pair(modul, gear_teeth, pinion_teeth, axis_angle=90, tooth_width, bore, pressure_angle = 20, helix_angle=0, together_built=true)
    worm
        worm(modul, thread_starts, length, bore, pressure_angle=20, lead_angle=10, together_built=true)
        worm_gear(modul, tooth_number, thread_starts, width, length, worm_bore, gear_bore, pressure_angle=20, lead_angle=0, optimized=true, together_built=true)
"""

from pathlib import Path
import numpy as np
from math import pi, cos, sin, tan, radians, degrees, ceil, asin, acos, sqrt
from typing import Tuple
from .pygears import involute_tooth

Vector2 = Tuple[float, float]
Vector3 = Tuple[float, float, float]


def rack_points(
    module=5,
    n_teeth=15,
    pressure_angle=radians(20),
    thickness=5,
    beta=0,
    head=0,
    clearance=0.25,
    properties_from_tool=False,
    add_endings=False,
    simplified=False,
    resolution=10,
):
    rack = involute_tooth.InvoluteRack(
        module,
        n_teeth,
        pressure_angle,
        thickness,
        beta,
        head,
        clearance,
        properties_from_tool,
        add_endings,
        simplified
    )

    return rack.points(resolution)


def involute_tooth_points(
    module=5,
    teeth=15,
    pressure_angle=radians(20),
    clearance=0.12,
    shift=0.5,
    beta=0.0,
    undercut=False,
    backlash=0.00,
    head=0.00,
    resolution=10,
):
    involute = involute_tooth.InvoluteTooth(
        m=module,
        z=teeth,
        pressure_angle=pressure_angle,
        clearance=clearance,
        shift=shift,
        beta=beta,
        undercut=undercut,
        backlash=backlash,
        head=head,
        properties_from_tool=False
    )

    # undercut_points = involute.undercut_points(resolution)
    # involute_points = involute.involute_points(resolution)

    points = involute.points(resolution)
    return points


if __name__ == "__main__":
    rack_points()
