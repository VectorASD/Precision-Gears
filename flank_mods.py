
from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class FlankMod:
    name: str
    type: str  # Must match enum
    prop_names: Dict[str, str]


class Crowing(FlankMod):
    name = "Crowning"
    type = "CROWNING"
    prop_names = {
        "flank_mod_param_a": "Amount",
        # "flank_mod_param_b": "Bias",
    }


class EndRelief(FlankMod):
    name = "End Relief"
    type = "END_RELIEF"
    prop_names = {
        "flank_mod_param_a": "Amount",
        "flank_mod_param_b": "Distance",
    }


def get(of_type: str) -> Union[FlankMod, None]:
    flank_mods = list(FlankMod.__subclasses__())
    return next((mod for mod in flank_mods if mod.type == of_type), None)


bores = list(FlankMod.__subclasses__())
