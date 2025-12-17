# TODO: This is all such a god damn mess
# TODO: Error in switch between rigging modes, missing distance attrib?


from __future__ import division, annotations
import math
from collections import deque
from dataclasses import dataclass, field
from itertools import chain, count
from numpy import sign
from typing import (
    TYPE_CHECKING,
    Any,
    Set,
    Generator,
    List,
    Tuple,
    Dict,
    TYPE_CHECKING,
    Union,
)
import logging
logger = logging.getLogger(f"{__name__}.rigging")
logger.addHandler(logging.StreamHandler())

import bpy
import bmesh
from bpy.types import Context, Object, GeometryNodeTree, GeometryNode
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import (
    consts,
    gear_math,
    bpy_helpers,
)
from .consts import GEAR_PROPS_ALIAS

if TYPE_CHECKING:
    from .props import PrecisionGearsProps, PrecisionGearsPreferences


def get_nodes_sans_output_link(tree: GeometryNodeTree):
    """ Get node with output sockets that have no output links """
    for node in tree.nodes:
        if not node.outputs:
            continue
        output_links = list(chain.from_iterable(out.links for out in node.outputs))
        if not output_links:
            yield node


def unlink_node_inputs(node: GeometryNode):
    """ remove all links connected to node's inputs """
    tree: GeometryNodeTree = node.id_data
    links = chain.from_iterable(input.links for input in node.inputs)
    for link in links:
        tree.links.remove(link)
    return node


def unlink_node_outputs(node: GeometryNode):
    """ remove all links connected to node's outputs """
    tree: GeometryNodeTree = node.id_data
    links = chain.from_iterable(output.links for output in node.outputs)
    for link in links:
        tree.links.remove(link)
    return node


def remove_node_links(node: GeometryNode):
    """ removal all links connected to node """
    unlink_node_inputs(node)
    unlink_node_outputs(node)
    return node


def get_rig_host(ctx: Context, node_group: GeometryNodeTree):
    logger.debug(f"Finding rig host for {node_group}")
    for o in ctx.scene.objects:
        if not o.modifiers:
            continue
        for modifier in o.modifiers:
            if modifier.type == "NODES":
                if modifier.node_group is node_group:
                    return o

    logger.debug(f"No rig host found for {node_group}")
    return None


@dataclass
class GearPair:
    driver: Object
    driven: Object
    driver_props: PrecisionGearsProps = field(init=False)
    driven_props: PrecisionGearsProps = field(init=False)
    driver_attribs: PrecisionGearsProps = field(init=False)
    driven_attribs: PrecisionGearsProps = field(init=False)

    # Rigging props
    driver_node: Union[bpy.types.GeometryNodeGroup, None] = None
    driven_node: Union[bpy.types.GeometryNodeGroup, None] = None

    def __post_init__(self):
        self.driver_props = _get_gear_props(self.driver)
        self.driven_props = _get_gear_props(self.driven)
        self.driver_attribs = _get_gear_attribs(self.driver)
        self.driven_attribs = _get_gear_attribs(self.driven)

        if self.driver_props.rig_tree:
            driver_tree = self.driver_props.rig_tree
            self.driver_node = _get_obj_rig_node(driver_tree, self.driver)
            self.driven_node = _get_obj_rig_node(driver_tree, self.driven)
            
            self.driven_node = driver_tree.nodes.get(self.driven_props.rigging_node, None)
            if self.driver_node == self.driven_node:
                self.driven_node = None
        
    def set_driven_rigging_props(self):
        self.driven_props.rig_tree = self.driver_props.rig_tree
        self.driven_props.root_driver = self.driver_props.root_driver
        self.driven_props.rig_host = self.driver_props.rig_host
    
    @property
    def compatible_rig_modes(self) -> Dict[str, set[str]]:
        return consts.VALID_RIG_DRIVERS[self.driven_props.gear_type]
    
    @property
    def rig_mode_is_compatible(self) -> bool:
        rig_mode = self.driven_props.rigging_mode
        driver_type = self.driver_props.gear_type
        return driver_type in self.compatible_rig_modes[rig_mode]
    
    def ensure_driven_rig_node(self, x_loc) -> GeometryNode:
        """ Ensure rig node and update x location"""
        tree = self.driver_node.id_data

        # Create rig node
        if self.driven_node is None:
            self.driven_node = tree.nodes.new("GeometryNodeGroup")
            self.driven_node.node_tree = bpy.data.node_groups.get(self.driven_props.rigging_mode)
            self.driven_node.inputs["Object"].default_value = self.driven
            self.driven_node.name = self.driven.name

        # Change rig node type
        if self.driver_node.node_tree.name != self.driven_props.rigging_mode:
            # Unlink node
            for output in self.driven_node.outputs:
                for link in output.links:
                    tree.links.remove(link)

            init_edit_mode = self.driven_props.editing
            self.driven_props.editing = True
            remove_node_links(self.driven_node)
            self.driven_node.node_tree = bpy.data.node_groups.get(self.driven_props.rigging_mode)
            self.driven_props.editing = init_edit_mode
            self.driven_node.name = self.driven.name
            self.driven_node.inputs["Object"].default_value = self.driven

        # Record node name and set location
        self.driven_props.rigging_node = self.driven_node.name
        self.driven_node.location = (x_loc, 0)
        return self.driven_node
     
    def connect_rig_node(self):
        rig_node_links = (
            ("Origin", "Origin"),
            ("Y", "Y"),
            ("Z", "Z"),
            ("State", "State"),
        )
        rigging_pair = (self.driver_node.node_tree.name, self.driven_node.node_tree.name)
        logger.debug(f"Connecting node {self.driver_node.name} to {self.driven_node.name}")
        connections = rig_node_links
        logger.debug(connections)
        tree = self.driven_node.id_data
        for from_socket, to_socket in connections:
            try:
                tree.links.new(
                    self.driver_node.outputs[from_socket],
                    self.driven_node.inputs[to_socket]
                )
            except Exception as e:
                logger.debug(e)
                pass
    
    def set_driven_rig_props(self):
        rig_prop_setter_type = RIG_NODE_INSTANTIATORS[self.driven_props.rigging_mode]
        prop_setter = rig_prop_setter_type(self.driver_node, self.driven_node)
        prop_setter.apply_node_parameters()


def get_rig_dependents(node: bpy.types.GeometryNodeGroup) -> List[GeometryNode]:
    return [link.to_node for link in node.outputs["State"].links]


def get_nodes_gear(node: bpy.types.GeometryNodeGroup):
    return node.inputs["Object"].default_value


def get_node_obj_refs(
    tree: bpy.types.GeometryNodeTree, obj: bpy.types.Object
) -> Generator[bpy.types.node, None, None]:
    """ Yield all nodes from tree that contain and input.default value of obj """
    input_is_obj = lambda _input: _input.default_value == obj
    for node in tree.nodes:
        has_ref = any(filter(input_is_obj, node.inputs))
        if has_ref:
            yield node


class RigNode:
    _connections = {}

    def __init__(
        self,
        driver_node: GeometryNode,
        node: GeometryNode,
    ):
        self.driver_node = driver_node 
        self.node = node
        self.driver_gear = self.driver_node.inputs["Object"].default_value
        self.driven_gear = self.node.inputs["Object"].default_value
        self.driver_props = _get_gear_props(self.driver_gear)
        self.driver_attribs = _get_gear_attribs(self.driver_gear)
        self.driven_props = _get_gear_props(self.driven_gear)
        self.driven_attribs = _get_gear_attribs(self.driven_gear)
        self.tree = self.driver_node.id_data

    def connect_to_driver(self):
        link_queue = self._connections[self.driver_node.node_tree.name]
        for from_socket, to_socket in link_queue:
            try:
                from_ = self.driver_node.outputs[from_socket]
                to_ = self.node.inputs[to_socket]
                self.tree.links.new(from_, to_)
            except Exception as e:
                print(e)
                pass

    def apply_node_parameters(self):
        driver_props = _get_gear_props(self.driver_gear)
        driven_props = _get_gear_props(self.driven_gear)
        type_pair = (driver_props.gear_type, driven_props.gear_type)
        try:
            rigging_calculator = self._rigging_calculators[type_pair]
            for prop, value in rigging_calculator().items():
                self.node.inputs[prop].default_value = value
        except KeyError as e:
            print(e)
        except Exception as e:
            print(e)
        
    @property
    def _rigging_calculators(self):
        return {}

    def add_material(self):
        try:
            self.node.inputs["Material"].default_value = self.driven_gear.active_material
        except Exception as e:
            pass


class MeshingRotRig(RigNode):
    node_group_name = "ROTATION"
    _connections = {
        "ROTATION": [
            ("World Loc", "Translation"),
            ("World Rot", "Rotation"),
            ("Orientation", "Inherit Orientation"),
            ("Angle Offset", "Inherited Angle Offset"),
            ("State", "State"),
        ],
        "COMPOUND": [
            ("World Loc", "Translation"),
            ("World Rot", "Rotation"),
            ("Orientation", "Inherit Orientation"),
            ("State", "State"),
        ],
    }

    @property
    def _rigging_calculators(self):
        # Driver, Driven
        return {
            ("SPUR", "SPUR"): self._spur_drive_spur,
            ("INTERNAL", "SPUR"): self._spur_drive_internal,
            ("SPUR", "INTERNAL"): self._spur_drive_internal,
            ("WORM", "SPUR"): self._worm_drive_spur,
            ("SPUR", "WORM"): self._spur_drive_worm,
            ("RACK", "SPUR"): self._rack_drive_spur,
            ("GT2GEAR", "GT2GEAR"): self._gt2_drive_gt2,
        }
    def _gt2_drive_gt2(self):
        rig_attribs = dict()
        props_a = _get_gear_props(self.driver_gear)
        attribs_a = _get_gear_attribs(self.driver_gear)
        props_b = _get_gear_props(self.driven_gear)
        attribs_b = _get_gear_attribs(self.driven_gear)
        scaler = 0.7

        distance = ((scaler * props_a.teeth) + (scaler * props_b.teeth)) / 2
        rig_attribs['Distance'] = distance
        rig_attribs['State Fac'] = (props_a.teeth / props_b.teeth)
        return rig_attribs

    def _spur_drive_spur(self):
        # TODO: Calculate distance with profile shift
        rig_attribs = dict()
        props_a = _get_gear_props(self.driver_gear)
        attribs_a = _get_gear_attribs(self.driver_gear)
        props_b = _get_gear_props(self.driven_gear)
        attribs_b = _get_gear_attribs(self.driven_gear)

        module = (props_a.module)
        teeth = (props_a.teeth, props_b.teeth)
        profile_shifts = (props_a.shift, props_b.shift)
        pressure_angle = props_a.pressure_angle
        distance = gear_math.spur_to_spur_distance(module, teeth, profile_shifts, pressure_angle)

        rig_attribs['Distance'] = distance
        rig_attribs['State Fac'] = -(props_a.teeth / props_b.teeth)
        return rig_attribs

    def _rack_drive_spur(self):
        rig_attribs = dict()
        rack_props = _get_gear_props(self.driver_gear)
        rack_attribs = _get_gear_attribs(self.driver_gear)
        spur_props = _get_gear_props(self.driven_gear)
        spur_attribs = _get_gear_attribs(self.driven_gear)

        rack_pitch_dist = rack_attribs["Rack Pitch Distance"]
        distance = gear_math.rack_to_spur_distance(spur_props.teeth, spur_props.module, rack_pitch_dist, rack_props.shift)
        rig_attribs["Distance"] = distance
        rig_attribs["State Fac"] = rack_attribs["Pitch Length"] / spur_attribs["Pitch Circumference"]
        return rig_attribs

    def _spur_drive_internal(self):
        # TODO: This seems slightly off?
        rig_attribs = dict()
        spur_props = _get_gear_props(self.driver_gear)
        spur_attribs = _get_gear_attribs(self.driver_gear)
        internal_props = _get_gear_props(self.driven_gear)
        internal_attribs = _get_gear_attribs(self.driven_gear)

        # Calculate Distance
        module = (spur_props.module)
        teeth = (spur_props.teeth, internal_props.teeth)
        profile_shifts = (spur_props.shift, internal_props.shift)
        pressure_angle = spur_props.pressure_angle
        distance = gear_math.spur_to_internal_distance(module, teeth, profile_shifts, pressure_angle)

        # rig_attribs['Distance'] = attribs_a["Pitch Radius"] - attribs_b["Pitch Radius"]
        rig_attribs['Distance'] = distance / 2
        rig_attribs['State Fac'] = spur_props.teeth / internal_props.teeth
        return rig_attribs

    def _worm_drive_spur(self):
        rig_attribs = dict()
        worm_props = _get_gear_props(self.driver_gear)
        worm_attribs = _get_gear_attribs(self.driver_gear)
        spur_props = _get_gear_props(self.driven_gear)
        spur_attribs = _get_gear_attribs(self.driven_gear)

        rig_attribs['Distance'] = (worm_attribs["Pitch Radius"] + spur_attribs["Pitch Radius"])
        rig_attribs['X Orbit'] = True

        # rig_attribs['State Fac'] = (spur_props.teeth / worm_props.teeth)
        rig_attribs['State Fac'] = -(worm_props.teeth / spur_props.teeth)

        if worm_props.reverse_pitch:
            rig_attribs['State Fac'] *= -1

        if worm_props.rigging_mode == "COMPOUND":
            rig_attribs['State Fac'] = -rig_attribs['State Fac']

        # if worm_props.rigging_mode == "COMPOUND":
        #     rig_attribs['State Fac'] = -rig_attribs['State Fac']

        return rig_attribs

    def _spur_drive_worm(self):
        rig_attribs = dict()
        spur_props = _get_gear_props(self.driver_gear)
        spur_attribs = _get_gear_attribs(self.driver_gear)
        worm_props = _get_gear_props(self.driven_gear)
        worm_attribs = _get_gear_attribs(self.driven_gear)

        rig_attribs["Distance"] = (spur_attribs["Pitch Radius"] + worm_attribs["Pitch Radius"])
        rig_attribs["Orientation"] = (0, math.radians(90), 0)

        # rig_attribs["Flip Out State"].default_value = worm_props.reverse_pitch

        # Determine State Fac
        rig_attribs["State Fac"] = (spur_props.teeth / worm_props.teeth)
        if worm_props.reverse_pitch:
            rig_attribs["State Fac"] = -rig_attribs["State Fac"]

        return rig_attribs
    

class CompoundRig(RigNode):
    node_group_name = "COMPOUND"

    _connections = {
        "ROTATION": [
            ("World Loc", "Translation"),
            ("World Rot", "Rotation"),
            ("Orientation", "Inherit Orientation"),
            ("Angle Offset", "Inherited Angle Offset"),
            ("State", "State"),
        ],
        "COMPOUND": [
            ("World Loc", "Translation"),
            ("World Rot", "Rotation"),
            ("Orientation", "Inherit Orientation"),
            ("State", "State"),
        ],
    }

    def apply_node_parameters(self):
        driver_props = _get_gear_props(self.driver_gear)
        driver_attribs = _get_gear_attribs(self.driver_gear)
        driven_props = _get_gear_props(self.driven_gear)
        driven_attribs = _get_gear_attribs(self.driven_gear)

        if driven_props.gear_type == "WORM":
            self.node.inputs["Out X as Z"].default_value = True
            # self.node.inputs["Flip In State"].default_value = True
            # self.node.inputs["Flip Out State"].default_value = driven_props.reverse_pitch:
        if driver_props.gear_type == "WORM":
            # self.node.inputs["Flip In State"].default_value = True
            self.node.inputs["In X as Z"].default_value = True

            if driver_props.driver is None or not driven_props.reverse_pitch:
                self.node.inputs["Flip In State"].default_value = False
            else:
                self.node.inputs["Flip In State"].default_value = True

            if driver_props.rigging_mode == "COMPOUND":
                if driven_props.reverse_pitch:
                    self.node.inputs["Flip In State"].default_value = False
                else: 
                    self.node.inputs["Flip In State"].default_value = True


class LinearRig(RigNode):
    node_group_name = "LINEAR"

    _connections = {
        "ROTATION": [
            ("World Loc", "Translation"),
            ("World Rot", "Rotation"),
            ("Orientation", "Inherit Orientation"),
            ("State", "State"),
        ],
    }

    @property
    def _rigging_calculators(self):
        return {
            ("SPUR", "RACK"): self._spur_drive_rack,
        }

    def _spur_drive_rack(self):
        rig_attribs = dict()
        spur_props = _get_gear_props(self.driver_gear)
        spur_attribs = _get_gear_attribs(self.driver_gear)
        rack_props = _get_gear_props(self.driven_gear)
        rack_attribs = _get_gear_attribs(self.driven_gear)

        # sum_pitch_distances = (spur_attribs["Pitch Radius"] + rack_attribs["Rack Pitch Distance"])
        # rig_attribs["Distance"] = sum_pitch_distances

        rack_pitch_dist = rack_attribs["Rack Pitch Distance"]
        distance = gear_math.rack_to_spur_distance(spur_props.teeth, spur_props.module, rack_pitch_dist, spur_props.shift)
        rig_attribs["Distance"] = distance
        rig_attribs["State Fac"] = spur_attribs["Pitch Circumference"] / rack_attribs["Pitch Length"]
        rig_attribs["Length"] = rack_attribs["Pitch Length"]
        rig_attribs["Orientation"] = (0, 0, math.pi)
        return rig_attribs


class BevelRig(RigNode):
    node_group_name = "BEVEL"

    @property
    def _rigging_calculators(self):
        return {
            ("BEVEL", "BEVEL"): self._bevel_drive_bevel,
        }

    def _bevel_drive_bevel(self):
        rig_attribs = dict()
        driver_props = _get_gear_props(self.driver_gear)
        driver_attribs = _get_gear_attribs(self.driver_gear)
        driven_props = _get_gear_props(self.driven_gear)
        driven_attribs = _get_gear_attribs(self.driven_gear)

        pitch_angle_sum = self.driver_props.pitch_angle + self.driven_props.pitch_angle
        rig_attribs["Sum Pitch Angles"] = pitch_angle_sum
        rig_attribs["Driver Apex Length"] = self.driver_attribs["apex"]
        rig_attribs["Apex"] = self.driven_attribs["apex"]
        rig_attribs["State Fac"] = -self.driver_props.teeth / self.driven_props.teeth
        return rig_attribs


RIG_NODE_INSTANTIATORS = {
    "ROTATION": MeshingRotRig,
    "LINEAR": LinearRig,
    "COMPOUND": CompoundRig,
    "BEVEL": BevelRig,
}


def _get_gear_props(gear: Object) -> PrecisionGearsProps:
    return getattr(gear, consts.GEAR_PROPS_ALIAS)


def _get_gear_attribs(gear: Object) -> Dict[str, Any]:
    return dict(gear.get(consts.GEAR_ATTRIBS_ALIAS))


def _get_all_rigged_gears(ctx: Context) -> Generator[Object, None, None]:
    """ Return all gear objects in a scene """
    def _is_gear(obj: Object) -> bool:
        props: PrecisionGearsProps = getattr(obj, consts.GEAR_PROPS_ALIAS)
        return props.is_gear and props.rigging_enabled
    return filter(_is_gear, ctx.scene.objects)


def _get_all_gears(ctx: Context) -> Generator[Object, None, None]:
    """ Return all gear objects in a scene """
    def _is_gear(obj: Object) -> bool:
        props: PrecisionGearsProps = getattr(obj, consts.GEAR_PROPS_ALIAS)
        return props.is_gear
    return filter(_is_gear, ctx.scene.objects)


def _get_all_drivers(ctx: Context):
    def _is_driver(obj: Object) -> bool:
        props: PrecisionGearsProps = getattr(obj, consts.GEAR_PROPS_ALIAS)
        return props.driver is None
    return filter(_is_driver, _get_all_rigged_gears(ctx))


def _ensure_no_duplicate_node_refs(ctx: Context):
    logger.debug("Ensuring no duplicate node references")
    nodes_reffed = set()
    for gear in _get_all_rigged_gears(ctx):
        props = _get_gear_props(gear)
        if props.rigging_node in nodes_reffed:
            props.rigging_node = ""
            continue
        nodes_reffed.add(props.rigging_node)


def _ensure_no_duplicate_drivers(ctx: Context):
    logger.debug("Ensuring no duplicate rig drivers")
    drivers = _get_all_drivers(ctx)
    used_rig_trees = set()
    for driver in drivers:
        props: PrecisionGearsProps = getattr(driver, consts.GEAR_PROPS_ALIAS)
        if props.rig_tree in used_rig_trees:
            # Driver is a duplicate, clear its props
            logger.debug(f"Found duplicate driver {driver}, clearing props")
            props.rig_tree = None
            props.root_driver = None
            props.rig_host = None
            props.rigging_node = ""
            continue
        used_rig_trees.add(props.rig_tree)


def _iter_chain_gears(ctx: Context, root_driver: Object) -> Generator[Tuple[Object, Object], None, None]:
    """ Yield chain gears as generator of (driver, driven) """
    unidentified_gears = set(_get_all_rigged_gears(ctx))
    chain_members = set((root_driver,))

    while chain_members:
        unidentified_gears.difference(chain_members)
        driver = chain_members.pop()

        for gear in unidentified_gears:
            gear_props: PrecisionGearsProps = getattr(gear, consts.GEAR_PROPS_ALIAS)
            if gear_props.driver is driver:
                yield driver, gear
                chain_members.add(gear)
        

def _circular_dependecy_exists(ctx: Context) -> bool:
    """
    Check for circular driver depedency using
    topological sorting(Kahn's algorithm)
    """
    logger.debug("Checking for circular rigging dependencies")

    scene_gears: List[Object] = []
    links: Dict[Object, Set[Object]] = {}
    test_queue = deque()

    # Collect iterables
    for obj in ctx.scene.objects:
        props = getattr(obj, consts.GEAR_PROPS_ALIAS)
        if not props.is_gear:
            continue
        scene_gears.append(obj)
        if props.driver is None:
            test_queue.append(obj)
        else:
            driven = links.setdefault(props.driver, set())
            driven.add(obj)

    while test_queue:
        driver = test_queue.pop()
        if driver in links.keys():
            for driven in links[driver]:
                test_queue.append(driven)
            links.pop(driver)
    if links != {}:
        logger.error("ERROR: Driver would cause circular dependency")
        return True
    return False


def _enable_rigging(gear_props):
    """ Enable rigging without triggering an update callback """
    gear_props.editing = True
    gear_props.rigging_enabled = True
    gear_props.editing = False


def _remove_tree_outputs(tree: GeometryNodeTree):
    """ Remove all links connected to GROUP_OUTPUT nodes """
    output_nodes = (n for n in tree.nodes if n.type == "GROUP_OUTPUT")
    inputs = chain.from_iterable(n.inputs for n in output_nodes)
    links = chain.from_iterable(socket.links for socket in inputs)
    for link in links:
        tree.links.remove(link)


def correct_intersection(ctx: Context, target: Object, recursive = True, from_root = False, samples = 10):
    """ Find the rig node value offset that results in the least intersection """
    logger.debug(f"Correcting rig intersection for {target}")
    target_props = _get_gear_props(target)
    tree = target_props.rig_tree
    rig_host = target_props.rig_host
    bool_node = tree.nodes.new("GeometryNodeMeshBoolean")
    bool_node.operation = "INTERSECT"

    if recursive or from_root:
        if from_root:
            target = _find_root_driver(target_props)
        targets = _iter_chain_gears(ctx, target)
    else:
        targets = [(target_props.driver, target),]

    for driver, driven in targets:
        # Clear tree outputs
        _remove_tree_outputs(tree)

        driver_node = get_rigging_node(driver)
        driven_node = get_rigging_node(driven)
        pair = GearPair(driver, driven, driver_node, driven_node)
        rig_prop = consts.RIG_NODE_INTERSECTION_PROP[pair.driven_props.gear_type]

        gear_types = (pair.driver_props.gear_type, pair.driven_props.gear_type)
        if "GT2GEAR" in gear_types:
            continue

        if rig_prop == "Offset":
            driven_attribs = _get_gear_attribs(pair.driven)
            test_step = (driven_attribs["Pitch"]) / samples
        else:
            test_step = (math.tau / pair.driven_props.teeth) / samples
        
        dg = ctx.evaluated_depsgraph_get()

        # Create driver bvh cache
        driver_link = tree.links.new(driver_node.outputs["Geometry"], tree.nodes["Group Output"].inputs["Geometry"])
        dg.update()
        evaled_rig = rig_host.evaluated_get(dg)
        mesh = evaled_rig.to_mesh()
        driver_bm = bmesh.new()
        driver_bm.from_mesh(mesh)
        driver_bvh = BVHTree.FromBMesh(driver_bm)
        driver_bm.free()
        evaled_rig.to_mesh_clear()
        tree.links.remove(driver_link)

        results = []
        driver_link = tree.links.new(driven_node.outputs["Geometry"], tree.nodes["Group Output"].inputs["Geometry"])

        driven_bm = bmesh.new()
        init_value = driven_node.inputs[rig_prop].default_value

        for i in range(samples):
            test_value = test_step * (i + 1)
            driven_node.inputs[rig_prop].default_value = init_value + test_value
            dg.update()
            evaled_rig = rig_host.evaluated_get(dg)
            mesh = evaled_rig.to_mesh()
            driven_bm.from_mesh(mesh)
            driven_bvh = BVHTree.FromBMesh(driven_bm)
            evaled_rig.to_mesh_clear()
            intersections = len(driver_bvh.overlap(driven_bvh))
            results.append((test_value, intersections))
            driven_bm.clear()

        driven_bm.free()

        # Set best value
        results.sort(key=lambda result: result[1])
        driven_node.inputs[rig_prop].default_value = init_value + results[0][0]

    # _remove_tree_outputs(tree)
    root_driver = _find_root_driver(target_props)
    chain_gears = set(chain.from_iterable(_iter_chain_gears(ctx, root_driver)))
    chain_nodes = map(get_rigging_node, chain_gears)
    join_and_output_nodes(chain_nodes)
    tree.nodes.remove(bool_node)


def _remove_node_groups_without_valid_driver():
    """ Remove all rigging node setups whose driver object has a driver and is thereforce invalid """
    logger.debug(f"Checking node groups without a driver")

    def _tree_has_root_driver_node(tree: GeometryNodeTree) -> bool:
        """ Identify addon node groups by presence of root driver node """
        return tree.nodes.get(consts.ROOT_DRIVER_NODE_NAME) is not None
    
    def _tree_has_state_controller(tree: GeometryNodeTree) -> bool:
        """ Identify addon node groups by presence of root driver node """
        state_controller = get_state_controller(tree) is not None
        return state_controller

    def _driver_is_invalid(tree: GeometryNodeTree):
        try:
            state_controller = get_state_controller(tree)
            root_node = state_controller.outputs["State"].links[0].to_node
            logger.debug(root_node)
            gear_ref = root_node.inputs["Object"].default_value
            if gear_ref is None:
                return True
            props: PrecisionGearsProps = getattr(gear_ref, consts.GEAR_PROPS_ALIAS)
            if props.rig_tree is not tree:
                return True
            if not props.rigging_enabled:
                return True
            return props.driver is not None
        except Exception as e:
            logger.debug(e)
            return True

    # rigging_trees = filter(_tree_has_root_driver_node, bpy.data.node_groups[:])
    rigging_trees = filter(_tree_has_state_controller, bpy.data.node_groups[:])
    invalid_trees = filter(_driver_is_invalid, rigging_trees)
    for tree in invalid_trees:
        logger.debug(f"Removing invalid node tree {tree}")
        bpy.data.node_groups.remove(tree)


def _remove_unused_node_hosts():
    logger.debug("Removing unused node hosts")
    def _has_unused_nodes_modifier(obj: Object) -> bool:
        for modifier in obj.modifiers:
            if modifier.type == "NODES":
                if modifier.node_group is None:
                    logger.debug(f"{obj} has unused nodes modifier")
                    return True
        return False

    def is_rig_host(o: Object):
        return _get_gear_props(o).is_rigging_host

    rig_objects = filter(is_rig_host, bpy.data.objects[:])
    invalid_hosts = filter(_has_unused_nodes_modifier, rig_objects)
    for host in invalid_hosts:
        logger.debug(f"{host} is not a valid rig host, removing")
        bpy.data.objects.remove(host)


def get_rigging_node(source: Union[Object, PrecisionGearsProps]):
    if isinstance(source, Object):
        source = _get_gear_props(source)
    try:
        return source.rig_tree.nodes.get(source.rigging_node)
    except Exception as e:
        print(e)
        return None


def is_rigging_node_type_correct(node: GeometryNode, props: PrecisionGearsProps) -> bool:
    return props.rigging_mode == node.node_tree.name


def has_correct_driver_connected(node: GeometryNode, props: PrecisionGearsProps) -> bool:
    try: 
        if props.driver is None:
            return True
        driver_node = node.inputs["State"].links[0].from_node
        return driver_node.inputs["Object"].default_value is props.driver
    except Exception as e:
        print(e)
        return False


def _create_rig_tree(host: Object) -> GeometryNodeTree:
    logger.debug(f"Creating new rig tree on {host}")
    tree = bpy.data.node_groups.new(host.name, "GeometryNodeTree")
    tree.nodes.new("NodeGroupOutput").name = "Group Output"
    if bpy.app.version[0] < 4:
        tree.outputs.new("NodeSocketGeometry", "Geometry")
    else:
        tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    modifier = host.modifiers.new(host.name, type="NODES")
    modifier.node_group = tree
    return tree


def join_and_output_nodes(nodes):
    new_links = []
    nodes = list(nodes)
    tree = nodes[0].id_data
    output_node = tree.nodes.get("Group Output")
    merge_node = tree.nodes.new("GeometryNodeJoinGeometry")

    # with open("C:/Users/VectorASD/AppData/Roaming/Blender Foundation/Blender/4.5/scripts/addons/Precision Gears/log.txt", "a") as file:
    #     file.write(f"~~~\nnodes: {nodes}\ntree: {tree}\noutput_node: {output_node}\nmerge_node: {merge_node}\n")
    # bpy.save = nodes, tree, output_node, merge_node

    # nodes       = [bpy.data.node_groups['Spur Gear_rig'].nodes["ROOT_DRIVER"]],
    # tree        = bpy.data.node_groups['Spur Gear_rig']
    # output_node = None                                                               WARNING! WARNING! WARNING!
    # merge_node  = bpy.data.node_groups['Spur Gear_rig'].nodes["Соединить геометрию"]

    """
>>> tree = bpy.data.node_groups['Spur Gear_rig'] # bpy.save[1]
>>> print(*tree.nodes, sep="\n")
<bpy_struct, NodeGroupOutput("Выходы группы") at 0x0000028040D0D188>
<bpy_struct, GeometryNodeGroup("ROOT_DRIVER") at 0x0000028040D0D788>
<bpy_struct, GeometryNodeGroup("GEAR_STATE_CONTROL") at 0x0000028040D0DA88>
<bpy_struct, GeometryNodeJoinGeometry("Соединить геометрию") at 0x0000028040D0F708>

>>> tree.nodes.get("Group Output")
>>> tree.nodes.get("Выходы группы")
bpy.data.node_groups['Spur Gear_rig'].nodes["Выходы группы"]
    """

    max_x = -999
    for node in nodes:
        max_x = max(node.location.x, max_x)
        new_links.append(tree.links.new(node.outputs["Geometry"], merge_node.inputs["Geometry"]))
    new_links.append(tree.links.new(merge_node.outputs["Geometry"], output_node.inputs["Geometry"]))

    # Move merge and out nodes to end of graph
    merge_node.location.x = max_x + 300
    output_node.location.x = max_x + 600
    return new_links


def remove_geom_merges(tree):
    for node in tree.nodes[:]:
        if node.type == "JOIN_GEOMETRY":
            tree.nodes.remove(node)


def _add_root_driver_nodes(tree: GeometryNodeTree, root_driver: Object):
    logger.debug(f"Adding root rig node for {root_driver}")
    props = _get_gear_props(root_driver)
    if props.gear_type in {"SPUR", "INTERNAL", "WORM", "BEVEL", "GT2GEAR"}:
        rig_node_group = "ROTATION"
    else:
        rig_node_group = "LINEAR"

    rig_node = tree.nodes.new("GeometryNodeGroup")
    rig_node.node_tree = bpy.data.node_groups[rig_node_group]
    rig_node.inputs["Object"].default_value = root_driver
    rig_node.name = consts.ROOT_DRIVER_NODE_NAME
    rig_node.label = consts.ROOT_DRIVER_NODE_NAME
    props.rigging_node = rig_node.name  # Assigned like this in case of name overlap
    attribs = _get_gear_attribs(root_driver)

    if props.gear_type == "WORM":
        rig_node.inputs["Orientation"].default_value = (0, math.pi / 2, 0)
        # rig_node.inputs["Flip Out State"].default_value = props.reverse_pitch
    elif props.gear_type == "RACK":
        rig_node.inputs["Length"].default_value = attribs["Pitch Length"]
        rig_node.inputs["Flip Y Out"].default_value = False
    return rig_node


def _clear_rigging_props(prop_grp: PrecisionGearsProps):
    logger.debug(f"Clearing rigging props on {prop_grp.id_data}")
    prop_grp.rig_tree = None
    prop_grp.root_driver = None
    prop_grp.rig_host = None
    prop_grp.rigging_node = ""


def _remove_unused_rig_nodes():
    logger.debug(f"Removing unused rigging nodes")
    rig_node_names = set(consts.RIGGING_NODE_GROUPS.keys())
    for tree in bpy.data.node_groups:
        if tree.name in rig_node_names:
            continue

        group_nodes = (node for node in tree.nodes if node.type == "GROUP")
        rig_nodes = (node for node in group_nodes if node.node_tree.name in rig_node_names)
        for node in rig_nodes:
            geo_output = node.outputs.get("Geometry")
            if geo_output is None:
                continue
            if not geo_output.links:
                print(geo_output.links)
                logger.debug(f"{node.name} in tree {tree.name} has no outputs, removing")
                tree.nodes.remove(node)


def update_rigging(gear_props: PrecisionGearsProps, ctx: Context):
    # TOOD: write a getter for getting addon preferences
    addon_prefs: PrecisionGearsPreferences = bpy.context.preferences.addons[__package__].preferences
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(getattr(logging, addon_prefs.debug_level))
    logger.debug(f"Running rigging update for {gear_props.id_data}")

    if gear_props.editing:  # Bypass update
        logger.debug(f"Bypassing rigging update for {gear_props.id_data}")
        return None

    gear_props.editing = True

    try:
        # Handle root becoming dependency
        if gear_props.driver is not None and 'ROOT' in gear_props.rigging_node:
            logger.debug(f"Clearing rigging props for {gear_props.id_data.name}")
            _clear_rigging_props(gear_props)

        _ensure_no_duplicate_drivers(ctx)
        # _ensure_no_duplicate_node_refs(ctx)
        if not gear_props.rigging_enabled:
            logger.debug(f"Rigging disabled for {gear_props.id_data}, clearing props")
            _clear_rigging_props(gear_props)
        if not gear_props.driver is None:
            _ensure_valid_driver(ctx, gear_props)
        
        # Ensure rig tree foundations present
        ensure_rig_node_groups_linked()
        root_driver = _find_root_driver(gear_props)
        root_driver_props = _get_gear_props(root_driver)

        caller_is_root = gear_props.id_data is root_driver
        if not caller_is_root:
            logger.debug(f"Enabling rigging on {root_driver_props.id_data}")
            _enable_rigging(root_driver_props)
        if not root_driver_props.rig_host:
            logger.debug(f"{root_driver_props.id_data} has no rig host")
            root_driver_props.rig_host = _create_rigging_host(ctx, root_driver.name)
        if not root_driver_props.rig_tree:
            logger.debug(f"{root_driver_props.id_data} has no rig tree")
            root_driver_props.rig_tree = _create_rig_tree(root_driver_props.rig_host)
        root_node = get_rigging_node(root_driver)
        if root_node is None:
            logger.debug(f"{root_driver_props.id_data} has no root node")
            root_node = _add_root_driver_nodes(root_driver_props.rig_tree, root_driver)
            root_state_controller = _add_root_state_controller(root_node)

        # Remove join node
        remove_geom_merges(root_driver_props.rig_tree)
        x_loc = count(0, 300)
        root_node.location.x = next(x_loc)
        rig_nodes = [root_node]

        # Handle rack gear prop updating
        if caller_is_root and gear_props.gear_type == "RACK":
            gear_attribs = _get_gear_attribs(gear_props.id_data)
            root_node.inputs["Length"].default_value  = gear_attribs["Pitch Length"]

        for driver, driven in _iter_chain_gears(ctx, root_driver):
            pair = GearPair(driver, driven)
            pair.set_driven_rigging_props()
            if not pair.rig_mode_is_compatible:
                continue

            rig_node = pair.ensure_driven_rig_node(next(x_loc))
            rig_nodes.append(rig_node)

            pair.connect_rig_node()
            pair.set_driven_rig_props()

        join_and_output_nodes(rig_nodes)

    except ValueError as e:
        logger.error(e)
        bpy_helpers.pop_error(str(e))
    except Exception as e:
        logger.error(e)
    finally:
        gear_props.refresh_rig = False
        gear_props.editing = False

    # Cleanup
    _remove_node_groups_without_valid_driver()
    _remove_unused_node_hosts()
    _remove_unused_rig_nodes()
    return None


def _add_root_state_controller(root_driver_node: GeometryNode):
    """ Create a instance of the root state controller node group and link to root driver """
    tree = root_driver_node.id_data
    logger.debug(f"Adding state controller to {root_driver_node}, in {tree}")

    # Instantiate node
    state_node = tree.nodes.new("GeometryNodeGroup")
    state_node.location = root_driver_node.location + Vector((-400, 0))
    state_node.name = consts.ROOT_DRIVER_STATE_NODE_NAME
    state_node.node_tree = bpy.data.node_groups[consts.ROOT_DRIVER_STATE_NODE_NAME]

    # Link to root driver
    tree.links.new(state_node.outputs["State"], root_driver_node.inputs["State"])

    # Return the node
    return state_node


def get_state_controller(node_tree: GeometryNodeTree):
    """ Return state controller node, or none if no state controller exists """
    # logger.debug(f"Finding state controller in {node_tree}")
    group_nodes = (node for node in node_tree.nodes if node.type == "GROUP")
    for node in group_nodes:
        if node.node_tree is bpy.data.node_groups["GEAR_STATE_CONTROL"]:
            return node
    return None


def _get_root_state_controller(tree: GeometryNodeTree) -> Union[GeometryNode, None]:
    """
    Return node named consts.ROOT_DRIVER_STATE_NODE_NAME from tree, or None it doesn't exist
    NOTE: I don't know if this is used anymore
    """
    return tree.nodes.get(consts.ROOT_DRIVER_STATE_NODE_NAME, None)


def _ensure_valid_driver(ctx: Context, props: PrecisionGearsProps) -> Union[str, bool]:
    """ Test rigging target is valid """
    logger.debug(f"Ensuring valid driver for {props.id_data}")
    compatible_modes = consts.VALID_RIG_DRIVERS[props.gear_type]

    # def driver_valid_valid_for_rig():
    def rig_mode_is_compatible():
        return props.rigging_mode in compatible_modes.keys()

    def driver_is_compatible_type():
        compatible_drivers = compatible_modes[props.rigging_mode]
        return _get_gear_props(props.driver).gear_type in compatible_drivers
    
    def referenced_object_in_scene():
        return ctx.scene.objects.get(props.driver.name) is not None

    error = None

    driver_is_self = props.driver is props.id_data
    if props.driver is not None:
        driver_props = _get_gear_props(props.driver)
        if driver_is_self:
            error = "Driver cannot be self"
        elif not driver_props.is_gear:
            error = "Driver is not a gear"
        elif _circular_dependecy_exists(ctx):
            error = f"Driver, {props.driver.name} would create circular dependency"
        elif not referenced_object_in_scene():
            error = f"Driver, {props.driver.name} is not in scene."
        elif not rig_mode_is_compatible():
            error = f"Rig mode, {props.rigging_mode.title()}, is not compatible between {props.gear_type.title()} and {driver_props.gear_type.title()}"
        elif not driver_is_compatible_type():
            error = f"Driver type, {props.rigging_mode.title()}, is not compatible with {driver_props.gear_type.title()}."

    if error is not None:
        props.driver = None
        raise ValueError(error)
    return True


def _has_valid_driver(ctx: Context, props: PrecisionGearsProps) -> Union[str, bool]:
    """ Test rigging target is valid """
    # TODO: Ensure no longer necessary and remove if so
    compatible_modes = consts.VALID_RIG_DRIVERS[props.gear_type]

    def rig_mode_is_compatible():
        return props.rigging_mode in compatible_modes.keys()

    def driver_is_compatible_type():
        compatible_drivers = compatible_modes[props.rigging_mode]
        return _get_gear_props(props.driver).gear_type in compatible_drivers

    def _tests() -> Generator[bool, None, None]:
        """Yield result of each test"""
        yield props.driver is not None  # Driver property has a value
        yield ctx.scene.objects.get(props.driver.name) is not None  # Object referenced in driver prop exists
        yield props.driver is not props.id_data  # Driver is not self
        yield rig_mode_is_compatible()
        yield driver_is_compatible_type()

    return all(_tests())


def _get_rig_tree(ctx: Context, root_driver_props: PrecisionGearsProps) -> GeometryNodeTree:
    """
    Find the rigging node group, creating if necessary
    TODO: Do I still need this?
    """ 
    if not root_driver_props.rig_tree:
        root_driver_name = root_driver_props.id_data.name
        rigging_host = _create_rigging_host(ctx, root_driver_name)
        root_driver_props.rig_tree = _create_rig_tree_setup(
            rigging_host, ctx, f"{root_driver_name}_rig"
        )
    return root_driver_props.rig_tree


def _create_rigging_host(ctx: Context, name: str) -> Object:
    """ Create and return object on which to spawn rigging setup """
    host_name = name + "_rig"
    logger.debug(f"Creating new rigging host {host_name}")
    mesh = bpy.data.meshes.new(host_name)
    obj = bpy.data.objects.new(host_name, object_data=mesh)
    gear_props = _get_gear_props(obj)
    gear_props.is_rigging_host = True
    ctx.collection.objects.link(obj)
    return obj


def _create_rig_tree_setup(obj: Object, ctx: Context, name: str):
    """ Add new modifier node group """
    # TODO: Rename this
    logger.debug(f"Adding rig tree to {obj}")
    node_group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    modifier = obj.modifiers.new(name, type="NODES")
    modifier.node_group = node_group
    return node_group


def _find_root_driver(gear_props: PrecisionGearsProps) -> Object:
    """ Find and return root driver object """
    logger.debug(f"Finding root driver of: {gear_props.id_data}")
    props = gear_props
    while props.driver is not None:
        next_gear = props.driver
        props = getattr(next_gear, consts.GEAR_PROPS_ALIAS)

    logger.debug(f"Root driver of {gear_props.id_data} is {props.id_data}")
    return props.id_data


def _get_obj_rig_node(tree: GeometryNodeTree, obj: Object):
    """ Find and returns rigging node that has obj set as Object input """
    def _is_obj_rig_node(node: GeometryNode):
        obj_input = node.inputs.get("Object")
        yield obj_input is not None
        yield obj_input.default_value is obj

    for node in tree.nodes:
        if all(_is_obj_rig_node(node)):
            return node
    return None


def ensure_rig_node_groups_linked():
    """ Append rigging nodes groups """
    logger.debug("Ensuring node groups are linked")
    node_groups = {}
    filepath = str(consts.RIGGING_NODE_GROUPS_PATH)
    with bpy.data.libraries.load(filepath, link=True) as (data_from, data_to):
        missing_node_groups = []
        for key, group_name in consts.RIGGING_NODE_GROUPS.items():
            if group_name not in bpy.data.node_groups.keys():
                missing_node_groups.append(group_name)

        data_to.node_groups = [
            group for group in data_from.node_groups
            if group in missing_node_groups
        ]

    for key, group_name in consts.RIGGING_NODE_GROUPS.items():
        node_groups[key] = bpy.data.node_groups.get(group_name)
    return node_groups


def make_driven_compatible(gear: Object) -> None:
    """Make a gear compatible with its driver"""
    props: PrecisionGearsProps = getattr(gear, GEAR_PROPS_ALIAS)
    driver: Object = props.driver
    driver_props: PrecisionGearsProps = getattr(driver, GEAR_PROPS_ALIAS)

    try:
        props.editing = True
        props.module = driver_props.module
        props.pressure_angle = driver_props.pressure_angle

        if driver_props.gear_type in ("SPUR", "INTERNAL", "RACK"):
            props.height = driver_props.height
            props.shift = driver_props.shift
            props.backlash = driver_props.backlash
            props.herringbone = driver_props.herringbone
            props.undercut = driver_props.undercut
            if props.gear_type in ("SPUR",):
                props.helix_angle = -driver_props.helix_angle
            else:
                props.helix_angle = driver_props.helix_angle
            props.z_resolution = driver_props.z_resolution
            if props.gear_type == "WORM":
                props.z_resolution = 64
        if driver_props.gear_type == "BEVEL":
            # Use driven gear's pitch angle to caclulcate appropriate tooth count
            driver_pitch_radius = (driver_props.teeth * driver_props.module) / 2
            driver_apex = driver_props.pitch_radius / math.tan(driver_pitch_radius)
            cone_distance = math.cos(driver_props.pitch_angle) * driver_apex
            pitch_diam = math.sin(props.pitch_angle) * cone_distance
            props.teeth = props.module / pitch_diam 

            props.helix_angle = -driver_props.helix_angle
            props.z_resolution = driver_props.z_resolution
            props.clearance = driver_props.clearance
            props.length = driver_props.length
            props.complexity = driver_props.complexity
        if driver_props.gear_type == "WORM":
            lead_angle = math.atan((driver_props.teeth * driver_props.module) / driver_props.diameter)
            props.helix_angle = lead_angle
            if driver_props.reverse_pitch:
                props.helix_angle = -props.helix_angle 
            props.module = driver_props.module
            props.pressure_angle = driver_props.pressure_angle
            props.tooth_mod_enabled = True
            props.worm_cut = props.module / 2
            props.herringbone = False
            props.shift = 0.0
    except Exception as e:
        print(e)
    finally:
        props.editing = False
        props.module = props.module  # Trigger mesh update

