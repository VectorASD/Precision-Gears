"""
TODO: disc cache mechanism. Either json or pickle
"""
import asyncio
from functools import partial, cache
from pathlib import Path
from itertools import count
from csv import DictReader, DictWriter
import pickle
from typing import Generator, Dict, Any, Iterable, Tuple, List
from ast import literal_eval

from bpy.path import clean_name

from . import consts

Preset = Dict[str, Any]
PresetCollection = Dict[str, Preset]
BpyEnum = List[Tuple[str, str, str, int]]
preset_categories_enum = []
_CSV_REQUIRED_FIELDS = {"preset_name", "thumb"}

def ensure_thumbnail_values(target_csv: Path):
    print(f'ensuring thumbnail values for {target_csv}')
    new_csv_fname = target_csv.with_suffix(".temp")
    with open(target_csv, "r", newline="") as csv_file:
        reader = DictReader(csv_file)
        fieldnames=reader.fieldnames

        with open(new_csv_fname, "w", newline="") as new_csv:
            writer = DictWriter(new_csv, fieldnames=fieldnames)
            writer.writeheader()
            for entry in reader:
                entry["thumb"] = clean_name(entry["preset_name"])
                writer.writerow(entry)

    target_csv.unlink()
    new_csv_fname.rename(target_csv)
    return target_csv

def validate_csv(csv: Path) -> bool:
    """
    Validate csv, ensuring fieldnames and calculating thumbnail filenames
    Return False if missing preset_name field else True of validation
    """
    with open(csv, "r", newline="") as csv_file:
        reader = DictReader(csv_file)
        if reader.fieldnames is None:
            return False
        csv_fields = list(reader.fieldnames)
        missing_fields = list(_CSV_REQUIRED_FIELDS.difference(csv_fields))
        if "preset_name" in missing_fields:
            return False

    return True


def _csv_reader(csv_file: Path) -> Generator[Dict[str, str], None, None]:
    """Yield lines of file as dicts"""
    with open(csv_file, "r") as preset_file:
        reader = DictReader(preset_file)
        for entry in reader:
            yield entry


def _type_preset_line(entry: Dict[str, str]) -> Preset:
    """Apply correct types to preset values"""

    def _ensure_formatting(value):
        """Ensure correct case (title) for None and bool types"""
        if isinstance(value, list):
            value = value[0]
        if value.lower() in {"false", "true", "none"}:
            return value.title()
        return value

    typed = {}
    formatted = {key: _ensure_formatting(value) for key, value in entry.items()}
    for key, value in formatted.items():
        try:
            evaled = literal_eval(value)
        except Exception as e:
            evaled = value
        typed.update({key: evaled})
    return typed



def _read_presets_file(csv_file: Path) -> Dict[str, Preset]:
    """
    Read csv files and return as dict of 'preset_name': preset
    Preset values are correctly typed
    """
    # Get presets
    untyped_presets = _csv_reader(csv_file)
    typed_presets = map(_type_preset_line, untyped_presets)
    return {preset["preset_name"]: preset for preset in typed_presets}


def get_default_preset(gear_type: str):
    preset_path = consts.PRESETS_DIR / gear_type / "Default.csv"
    try:
        return _read_presets_file(preset_path)['Default']
    except FileNotFoundError:
        print(f"Defaults file: {preset_path}, doesn't exist")
        return {}


def get_presets(gear_type: str):
    """
    Return parse all csv files in (consts.PRESETS_DIR / gear_type)
    and return a dict of presets with correctly typed values
    """
    csv_paths = (consts.PRESETS_DIR / gear_type).glob("*.csv")
    presets = {}
    for csv_path in csv_paths:
        file_presets = _read_presets_file(csv_path)
        presets = presets | file_presets
    return presets


def get_preset_by_name(name: str, gear_type: str):
    """
    Return typed dict of named preset of gear_type 
    """
    presets = get_presets(gear_type)
    return presets.get(name)


def get_preset_names(gear_type: str):
    """ Return a list of preset names for gear_type """
    csv_paths = (consts.PRESETS_DIR / gear_type).glob("*.csv")
    preset_names = []
    for csv_path in csv_paths:
        # ensure_thumbnail_values(csv_path)
        for preset in _csv_reader(csv_path):
            preset_names.append(preset["preset_name"])
    return preset_names


def get_presets_enum(gear_type: str):
    preset_names = get_preset_names(gear_type)
    enum_index = count()
    enum = []
    enum.append(("None", "None", "None", next(enum_index)))
    try:
        preset_names.remove("Default")
        enum.append(("Default", "Default", "Default", next(enum_index)))
    except ValueError:
        pass
    for name in preset_names:
        enum.append((name, name, name, next(enum_index)))
    return enum

def apply_preset(prop_grp, preset_name, blocklist = None):
    preset = get_preset_by_name(preset_name, prop_grp.gear_type)
    if blocklist is not None:
        for key in blocklist:
            try:
                preset.pop(key)
            except KeyError:
                continue
    for prop, value in preset.items():
        try:
            setattr(prop_grp, prop, value)
        except Exception as e:
            # print(e)
            continue


async def generate_thumbnails() -> Path:
    """Run an instance of Blender in an asyc subprocess and render preset thumbnails with it"""

    async def handle_request(reader, writer, queue: asyncio.Queue):
        data = await reader.read(100)
        # message = data.decode()
        # addr = writer.get_extra_info("peername")
        # print(f"Received {message} from {addr}")
        # print("Responding to Job Request")
        task = await queue.get()
        writer.write(task)
        await writer.drain()

        # print("Close the connection")
        writer.close()
        queue.task_done()

    async def run_renderer(print_log: bool = True):
        ip = consts.THUMB_GEN_IP
        port = consts.THUMB_GEN_PORT
        template = consts.PRESET_THUMB_TEMPLATE
        script = consts.PRESET_THUMB_SCRIPT

        script_args = ["--", ip, str(port)]
        args = ["-b", template, "-P", script] + script_args
        proc = await asyncio.create_subprocess_exec(
            consts.BLENDER,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if print_log:
            if stdout:
                print(f"{stdout}\n{stdout.decode()}")
            if stderr:
                print(f"{stderr}\n{stderr.decode()}")

    async def main():
        job_queue = asyncio.Queue()
        resolution = int(consts.PRESET_THUMB_RESOLUTION)

        for gear_type in consts.GEAR_TYPES:
            presets = get_presets(gear_type)
            output_dir =  consts.PRESETS_DIR / gear_type
            job = (gear_type, presets, output_dir, resolution)
            await job_queue.put(pickle.dumps(job))

        # Write termination command
        terminate = pickle.dumps("TERMINATE")
        await job_queue.put(terminate)

        server_callback = partial(handle_request, queue=job_queue)
        ip = consts.THUMB_GEN_IP
        port = consts.THUMB_GEN_PORT
        job_server = await asyncio.start_server(server_callback, ip, port)

        addrs = ", ".join(str(sock.getsockname()) for sock in job_server.sockets)
        print(f"Running job server on {addrs}")
        async with job_server:
            await job_server.start_serving()
            await run_renderer(consts.DEBUG_PRINT_THUMB_RENDER_LOG)
            # await job_queue.join()

    # asyncio.run(main(), debug=True)
    print("Thumbnail rendering complete")
    await main()
