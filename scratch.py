from math import pi, acos, cos, sin, tan, radians, degrees
import numpy as np
from dataclasses import dataclass
from functools import cached_property

# Ref:
# khkgears.net/new/gear_knowledge/gear_technical_reference/gear_types_terminology.html

@dataclass
class GearParams:
    module: float = 5.0
    pressure_angle: float = 20.0  # Degrees
    n_teeth: float = 30.0
    ref_diameter: float = 150.0
    base_diameter: float = 140.95389
    tip_diameter: float = 160.0

    def __post_init__(self):
        self.pitch = pi * self.module
        self.addendum = self.module
        self.dedendum = 1.25 * self.module
        self.tooth_depth = 2.25 * self.module  # aka addendum + dedendum
        self.working_depth = 2.0 * self.module
        self.tip_root_clearance = 0.25 * self.module  # May vary, often 0.167 * m?
        self.dedendum_fillet_radius = 0.38 * self.module
        self.ref_radius = self.ref_diameter / 2
        self.base_radius = self.base_diameter / 2
        self.tip_radius = self.tip_diameter / 2

    def involute_curve(self, radius: float):
        angle = acos(self.base_radius / radius)
        inv_a = tan(angle) - angle
        x = radius * cos(inv_a)
        y = radius * sin(inv_a)
        return radius, degrees(angle), x, y


def involute_profile(base_radius: float, radius: float):
    angle = acos(base_radius / radius)
    inv_a = tan(angle) - angle
    x = radius * cos(inv_a)
    y = radius * sin(inv_a)
    return radius, degrees(angle), x, y


def check_gear_vals():
    gear = GearParams()
    test_radii = (70.47695, 72, 74, 76, 78, 80)

    base_radius = 140.95389 / 2
    for test_radius in test_radii:
        print(base_radius, test_radius)

if __name__ == "__main__":
    pass
