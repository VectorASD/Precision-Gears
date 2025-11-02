# context.area: NODE_EDITOR

from typing import Iterable
from itertools import count

import bpy
from bpy.types import Object, GeometryNodeTree


def geom_nodes_assembly(
    name: str,
    targets: Iterable[Object]
):
    tree: GeometryNodeTree = bpy.data.node_groups.new(name, type="GeometryNodeTree")

    # Create merge node
    merge = tree.nodes.new("GeometryNodeJoinGeometry")
    merge.location = (400, 0)

    y_locs = count(step=200)
    for target in targets:
        # Add reader and assign object
        y_val = next(y_locs)
        loader = tree.nodes.new("GeometryNodeObjectInfo")
        loader.location =  (0, y_val)
        loader.inputs["Object"].default_value = target

        # TODO: do I just need to set loader.transform_space ?????
        # Apply transform to geometry
        xform = tree.nodes.new("GeometryNodeTransform")
        xform.location =  (200, y_val)
        tree.links.new(loader.outputs["Location"], xform.inputs["Translation"])
        tree.links.new(loader.outputs["Rotation"], xform.inputs["Rotation"])
        tree.links.new(loader.outputs["Scale"], xform.inputs["Scale"])
        tree.links.new(loader.outputs["Geometry"], xform.inputs["Geometry"])

        # Link to join node
        tree.links.new(xform.outputs["Geometry"], merge.inputs["Geometry"])


    
    
    
    return tree


targets = list(bpy.data.objects)
geom_nodes_assembly("test_tree", targets)