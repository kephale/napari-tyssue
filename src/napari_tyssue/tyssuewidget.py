"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import logging
from typing import TYPE_CHECKING

# tyssue imports

# import pandas as pd
import numpy as np

# import json

from IPython.display import display, Image

import napari

# import vispy as vp

import tyssue
from tyssue import Sheet, History
from tyssue import config

from tyssue import SheetGeometry as geom
from tyssue.dynamics.sheet_vertex_model import SheetModel as basemodel
from tyssue.dynamics.apoptosis_model import SheetApoptosisModel as model
from tyssue.solvers.quasistatic import QSSolver

from tyssue.config.draw import sheet_spec
from tyssue.utils.utils import spec_updater

# from tyssue.draw.ipv_draw import _get_meshes
# from tyssue.draw.vispy_draw import sheet_view, face_visual, edge_visual

# from tyssue.draw import sheet_view, create_gif, browse_history
from tyssue.io.hdf5 import save_datasets, load_datasets

# napari imports

from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget

import napari
from napari.utils import progress

from threading import Thread

from superqt.utils import ensure_main_thread

LOGGER = logging.getLogger("napari_tyssue.TyssueWidget")


def face_mesh(sheet, coords, **face_draw_specs):
    """
    Creates a ipyvolume Mesh of the face polygons
    """
    Ne, Nf = sheet.Ne, sheet.Nf
    if callable(face_draw_specs["color"]):
        face_draw_specs["color"] = face_draw_specs["color"](sheet)

    if isinstance(face_draw_specs["color"], str):
        color = face_draw_specs["color"]

    elif hasattr(face_draw_specs["color"], "__len__"):
        color = _face_color_from_sequence(face_draw_specs, sheet)[:, :3]

    if "visible" in sheet.face_df.columns:
        edges = sheet.edge_df[
            sheet.upcast_face(sheet.face_df["visible"])
        ].index
        _sheet = get_sub_eptm(sheet, edges)
        if _sheet is not None:
            sheet = _sheet
            if isinstance(color, np.ndarray):
                faces = sheet.face_df["face_o"].values.astype(np.uint32)
                edges = edges.values.astype(np.uint32)
                indexer = np.concatenate([faces, edges + Nf, edges + Ne + Nf])
                color = color.take(indexer, axis=0)

    epsilon = face_draw_specs.get("epsilon", 0)
    up_srce = sheet.edge_df[["s" + c for c in coords]]
    up_trgt = sheet.edge_df[["t" + c for c in coords]]

    Ne, Nf = sheet.Ne, sheet.Nf

    if epsilon > 0:
        up_face = sheet.edge_df[["f" + c for c in coords]].values
        up_srce = (up_srce - up_face) * (1 - epsilon) + up_face
        up_trgt = (up_trgt - up_face) * (1 - epsilon) + up_face

    mesh_ = np.concatenate(
        [sheet.face_df[coords].values, up_srce.values, up_trgt.values]
    )

    triangles = np.vstack(
        [sheet.edge_df["face"], np.arange(Ne) + Nf, np.arange(Ne) + Ne + Nf]
    ).T.astype(dtype=np.uint32)

    color = np.linspace(0, 1, len(mesh_))

    mesh = (mesh_ * 10.0, triangles, color)
    return mesh


def _get_meshes(sheet, coords, draw_specs):

    meshes = []
    edge_spec = draw_specs["edge"]
    edge_spec["visible"] = False
    if edge_spec["visible"]:
        edges = edge_mesh(sheet, coords, **edge_spec)
        meshes.append(edges)
    else:
        edges = None

    face_spec = draw_specs["face"]
    face_spec["visible"] = True
    if face_spec["visible"]:
        faces = face_mesh(sheet, coords, **face_spec)
        meshes.append(faces)
    else:
        faces = None

    print("faces", faces)
    return meshes


class TyssueWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        # TODO create the layer that we will use for rendering
        self.layer = None

        # Flag used by simulation thread to change simulation state between timesteps
        self.running = False

        # Setup the UI
        # self._init_buttons()

    def _init_buttons(self):
        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.clicked.connect(self._on_start_click)

        self.stop_btn = QPushButton("Stop Simulation")
        self.stop_btn.clicked.connect(self._on_stop_click)

        self.export_btn = QPushButton("Export Simulation")
        self.export_btn.clicked.connect(self._on_export_click)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.start_btn)
        self.layout().addWidget(self.stop_btn)
        self.layout().addWidget(self.export_btn)

    def start_simulation(self):
        """
        OVERRIDE This method.

        This function will be run in a separate thread.
        The function should implement your simulation and
        viewer updates.
        """
        LOGGER.debug("TyssueWidget.start_simulation: not implemented")

    def _on_start_click(self):
        """
        This function is called when the start simulation button is clicked.
        It launches a new simulation thread.
        """
        LOGGER.info("start: napari has", len(self.viewer.layers), "layers")

        self.thread = Thread(target=self.start_simulation)
        self.thread.start()

    def _on_stop_click(self):
        LOGGER.info("stopping simulation")

        self.running = False
        self.thread.join()

        LOGGER.info("simulation stopped")

    def _on_export_click(self):
        LOGGER.info("export: not implemented")
