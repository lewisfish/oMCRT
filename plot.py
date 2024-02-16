import argparse
from collections import OrderedDict
import os
import re

import pyvista as pv
import numpy as np
import vtk

_types = {"float": np.float32, "double": np.float64,
          "int32": np.int32, "int64": np.int64,
          "uchar": np.ubyte}


def read_nrrd(file):
    with open(file, "rb") as fh:
        hdr = read_header(fh)
        data = read_data(fh, hdr)
    return data, hdr


def read_header(file):

    it = iter(file)
    magic_line = next(it)
    if hasattr(magic_line, "decode"):
        need_decode = True
        magic_line = magic_line.decode("ascii", "ignore")  # type: ignore[assignment]
        if not magic_line.startswith("NRRD"):  # type: ignore[arg-type]
            raise NotImplementedError

    header = OrderedDict()

    for line in it:
        if need_decode:
            line = line.decode("ascii", "ignore")  # type: ignore[assignment]

        line = line.rstrip()
        if line.startswith("#"):  # type: ignore[arg-type]
            continue
        elif line == "":
            break

        key, value = re.split(r"[:=?]", line, 1)  # type: ignore[type-var]
        key, value = key.strip(), value.strip()  # type: ignore[attr-defined]

        value = _get_value_type(key, value)

        header[key] = value

    return header


def _get_value_type(key, value):

    if key in ["dimension", "space dimension"]:
        return int(value)
    elif key in ["endian", "encoding"]:
        return value
    elif key in ["type"]:
        return _types[value]
    elif key in ["sizes"]:
        return [int(x) for x in value.split()]
    else:
        # ignore other metadata
        pass

def read_data(file, header):

    size = np.array(header["sizes"])
    dtype = header["type"]

    total_data_points = size.prod(dtype=np.int64)
    dtype_size = np.dtype(dtype).itemsize
    file.seek(-1 * dtype_size * total_data_points, os.SEEK_END)
    data = np.fromfile(file, dtype=dtype, sep="")
    data = data.reshape((size))
    return data

parser = argparse.ArgumentParser(prog="plot")
parser.add_argument("-f", "--file", type=str)

args = parser.parse_args()
filename = args.file
plotter = pv.Plotter()

mesh = pv.read("models/" + filename + ".obj")
actor = plotter.add_mesh(mesh, style='wireframe', opacity=.1)

voxels, hdr = read_nrrd(filename + ".nrrd")
grid = pv.UniformGrid()
grid.dimensions = np.array(voxels.shape) + 1
grid.spacing = (3./voxels.shape[0], 3./voxels.shape[1], 3./voxels.shape[2])
grid.origin = (-1.5, -1.5, -1.5)

fluence = voxels.flatten(order="C")
grid.cell_data["log(fluence)"] = np.log10(fluence, where=fluence>0 )  # Flatten the array!

thresh = grid.threshold(value=[0.0000001, np.max(grid.cell_data["log(fluence)"])], scalars="log(fluence)")
plotter.add_mesh_clip_plane(thresh, scalars="log(fluence)", assign_to_axis='z', interaction_event=vtk.vtkCommand.InteractionEvent)
plotter.camera.position = (0.0, 0.0, -10.0)
plotter.camera.up = (0.0, 1.0, 0.0)

## TODO -> fix camera movement to be relative to screen
def camDown(*args):
    cam = plotter.camera
    pos = cam.position
    fp = cam.focal_point
    plotter.camera.position = (pos[0], pos[1] - 1., pos[2])
    plotter.camera.focal_point = (fp[0], fp[1] - 1., fp[2])
    plotter.render()

def camUp(*args):
    cam = plotter.camera
    pos = cam.position
    fp = cam.focal_point
    plotter.camera.position = (pos[0], pos[1] + 1., pos[2])
    plotter.camera.focal_point = (fp[0], fp[1] + 1., fp[2])
    plotter.render()

def camLeft(*args):
    cam = plotter.camera
    pos = cam.position
    fp = cam.focal_point
    plotter.camera.position = (pos[0] - 1., pos[1], pos[2])
    plotter.camera.focal_point = (fp[0] - 1., fp[1], fp[2])
    plotter.render()

def camRight(*args):
    cam = plotter.camera
    pos = cam.position
    fp = cam.focal_point
    plotter.camera.position = (pos[0] + 1., pos[1], pos[2])
    plotter.camera.focal_point = (fp[0] + 1., fp[1], fp[2])
    plotter.render()

def setWireframe(*args):
    """toggle from wireframe to surface"""
    prop = actor.GetProperty()
    if prop.GetRepresentation() == 1:
        prop.SetRepresentationToSurface()
    else:
        prop.SetRepresentationToWireframe()
    plotter.render()

plotter.add_key_event("v", setWireframe)
plotter.add_key_event('Down', camDown)
plotter.add_key_event('Up', camUp)
plotter.add_key_event('Right', camRight)
plotter.add_key_event('Left', camLeft)

plotter.show()