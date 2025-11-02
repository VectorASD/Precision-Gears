from __future__ import division, annotations
from math import cos, sin, tan, pi, tau, atan, radians, degrees
from typing import Tuple
from operator import sub


# def involute_pitch_radius(module: float, teeth: int, shift: float = 0.0, offset: float = 0):
#     radius = (teeth * module) / 2
#     radius += module * shift
#     return radius + offset

##########################
# General Math Functions #
##########################
def involute(a):
    return tan(a) - a


def inverse_involute(inv):
    b = 0
    a = atan(0 + inv)

    while abs(a - b) > 0.00001:
        b = a
        a = atan(a + inv)
    return a


########################
# Distance calculators #
########################
def spur_to_spur_distance(
        module: float, teeth: Tuple[int, int], profile_shifts: Tuple[float, float], pressure_angle: float
):
    """ Calculate the distance between two profile shifted spur gears

    Args:
        module (float): Module
        teeth (Tuple[int, int]): Teeth counts
        profile_shifts (Tuple[float, float]): Profile shifts
        pressure_angle (float): Reference pressure angle
    Returns:
        (float): Distance between gears
    """

    sum_teeth = sum(teeth)

    def involute_of_gears() -> float:
        """ Return the involute of a pair of profile shifted spur gears """
        return 2 * tan(pressure_angle) * (sum(profile_shifts) / sum_teeth) + involute(pressure_angle)

    def distance_coefficient(working_pressure_angle: float):
        """ Return the center distance coefficient for a pair of spur gears """
        return (sum_teeth / 2) * (cos(pressure_angle) / cos(working_pressure_angle) - 1)

    involute_angle = involute_of_gears()
    working_pressure_angle = inverse_involute(involute_angle)
    distance_coeff = distance_coefficient(working_pressure_angle)
    distance = (sum_teeth / 2 + distance_coeff) * module
    return distance


def spur_to_internal_distance(module: float, teeth: Tuple[int, int], profile_shifts: Tuple[float, float], pressure_angle: float) -> float:
    """ Calculate the distance between profile shift supporting spur and internal gears """
    spur_teeth, internal_teeth = teeth
    spur_shift, internal_shift = profile_shifts
    teeth_dif = internal_teeth - spur_teeth
    shift_dif = internal_shift - spur_shift

    def involute_of_gears():
        return 2 * tan(pressure_angle) * (shift_dif / teeth_dif) + involute(pressure_angle)

    def distance_coefficient(working_pressure_angle):
        a = teeth_dif / 2
        b = cos(pressure_angle) / cos(working_pressure_angle) - 1
        return a * b

    # Involute of working pressure angle
    involute_angle = involute_of_gears()
    wpa = inverse_involute(involute_angle)
    cent_dist_coef = distance_coefficient(wpa)
    cent_dist = (teeth_dif + cent_dist_coef) * module
    return cent_dist


def rack_to_spur_distance(spur_teeth: int, module, rack_pitch_height, profile_shift) -> float:
    return (spur_teeth * module) / 2 + rack_pitch_height + profile_shift * module


if __name__ == "__main__":
    module = 3
    teeth = (16, 24)
    profile_shifts = (0, 0.516)
    pressure_angle = radians(20)
    distance = spur_to_internal_distance(module, teeth, profile_shifts, pressure_angle)
    print(distance)