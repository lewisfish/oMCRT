from collections import OrderedDict
import os
import re

import pyvista as pv
import numpy as np
import vtk
# from skimage.measure import marching_cubes


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
        pass
        # raise NotImplementedError


def read_data(file, header):

    size = np.array(header["sizes"])
    dtype = header["type"]

    total_data_points = size.prod(dtype=np.int64)
    dtype_size = np.dtype(dtype).itemsize
    file.seek(-1 * dtype_size * total_data_points, os.SEEK_END)
    data = np.fromfile(file, dtype=dtype, sep="")
    data = data.reshape((size))
    return data

plotter = pv.Plotter()


mesh = pv.read("models/sphere.obj")
plotter.add_mesh(mesh, style='wireframe', opacity=.1)

voxels, hdr = read_nrrd("sphere.nrrd")
grid = pv.UniformGrid()
grid.dimensions = np.array(voxels.shape) + 1
grid.spacing = (0.03, 0.03, 0.03)
grid.origin = (-1.5, -1.5, -1.5)

fluence = voxels.flatten(order="C")
grid.cell_data["log(fluence)"] = np.log10(fluence, where=fluence>0 )  # Flatten the array!

thresh = grid.threshold(value=[1, np.max(grid.cell_data["log(fluence)"])], scalars="log(fluence)")
plotter.add_mesh_clip_plane(thresh, scalars="log(fluence)", assign_to_axis='z', interaction_event=vtk.vtkCommand.InteractionEvent)
plotter.camera.position = (0.0, 0.0, -10.0)
plotter.camera.up = (0.0, 1.0, 0.0)

plotter.show()