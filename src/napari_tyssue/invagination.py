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
# import numpy as np
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

from tyssue import Sheet, config
from tyssue.io import hdf5

# from tyssue.draw import sheet_view
from tyssue.dynamics import effectors, model_factory

from tyssue.generation import ellipsoid_sheet
from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.sheet.delamination_events import constriction

from tyssue.solvers.quasistatic import QSSolver
from tyssue.geometry.sheet_geometry import EllipsoidGeometry as geom

## The invagination module in this repository provides defintions
## specific to mesoderm invagination

from invagination.ellipsoid import initiate_ellipsoid, define_mesoderm
from invagination.delamination import delamination_process, constriction_rate

# from invagination.plots import mesoderm_position
from invagination.ellipsoid import VitellineElasticity, RadialTension

# napari imports

from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget

import napari
from napari.utils import progress

from threading import Thread

from superqt.utils import ensure_main_thread

LOGGER = logging.getLogger("napari_tyssue.Invagination")

from .tyssuewidget import TyssueWidget, _get_meshes


# This widget wraps the invagination demo from tyssue.
# https://github.com/DamCB/invagination/blob/master/notebooks/SmallEllipsoidInvagination.ipynb
class InvaginationWidget(TyssueWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

        # tyssue model init

        # The history stores simulation outputs
        self.history = None

        # Current timestep
        self.t = 0

        # This is the stop time of the simulation
        self.stop = 100

        self.contractility_rate = 2
        self.critical_area = 5
        self.radial_tension = 40

        self.settings = {
            "contract_rate": self.contractility_rate,
            "critical_area": self.critical_area,
            "radial_tension": self.radial_tension,
            "nb_iteration": 10,
            "contract_neighbors": True,
            "contract_span": 1,
            "geom": geom,
        }

        self.specs = {
            "vert": {
                "height": 0,
                "basal_shift": 0,
                "delta_rho": 30,
                "vitelline_K": 280.0,
                "radial_tension": 0,
            },
            "face": {
                "contractility": 1.12,
                "prefered_area": 22,
                "area_elasticity": 1,
                "surface_tension": 10.0,
            },
            "edge": {
                "line_tension": 0.0,
            },
            "settings": {
                "abc": [12, 12, 21.0],  # Ellipsoid axes
                "geometry": "cylindrical",
                "height_axis": "z",
                "vitelline_space": 0.2,
                "threshold_length": 1e-3,
            },
        }

        # Add model parameters for config

        # Setup the UI
        self._init_buttons()

        # Add a new callback for the timeslider

    def start_simulation(self):

        sheet = ellipsoid_sheet(*self.specs["settings"]["abc"], 13)
        print(f"The sheet has {sheet.Nf} vertices")
        sheet.update_specs(self.specs)

        geom.update_all(sheet)

        def draw_specs(sheet):
            specs = {
                "edge": {"visible": True, "color": sheet.vert_df.y},
                "face": {
                    "visible": True,
                    "color": sheet.face_df.area,
                    "colormap": "Blues",
                },
            }
            return specs

        model = model_factory(
            [
                RadialTension,
                VitellineElasticity,
                effectors.FaceContractility,
                effectors.FaceAreaElasticity,
                effectors.LumenVolumeElasticity,
            ]
        )

        print("Our model has the following elements :")
        print("\t", *model.labels, sep="\n\t")

        # Modify some initial values
        sheet.face_df["prefered_area"] = sheet.face_df["area"].mean()
        sheet.settings["lumen_prefered_vol"] = 12666
        sheet.settings["lumen_vol"] = 11626
        sheet.settings["lumen_vol_elasticity"] = 1.0e-3

        geom.update_all(sheet)

        # Gradient descent

        solver_kw = {
            "method": "L-BFGS-B",
            "options": {"ftol": 1e-8, "gtol": 1e-8},
        }

        solver = QSSolver()
        res = solver.find_energy_min(sheet, geom, model, **solver_kw)

        print(res.message)
        # fig, ax = sheet_view(sheet, coords=list("zx"), mode="quick")

        # Define ovoid mesoderm
        define_mesoderm(sheet, a=15, b=6.0)

        mesoderm = sheet.face_df[sheet.face_df.is_mesoderm].index
        delaminating_cells = sheet.face_df[sheet.face_df["is_mesoderm"]].index
        sheet.face_df["is_relaxation"] = False
        print("number of apoptotic cells: {}".format(delaminating_cells.size))
        # fig, axes = mesoderm_position(sheet, delaminating_cells)

        sheet.face_df["id"] = sheet.face_df.index.values
        sheet.settings["delamination"] = self.settings

        # Run the simulation
        self.t = 0

        self.history = History(sheet)
        self.running = True

        self.viewer.dims.ndisplay = 3

        delaminating_cells = []
        # Initiate manager
        manager = EventManager("face", logfile="manager_log.txt")
        sheet.face_df["enter_in_process"] = 0

        # Add all cells in constriction process
        for f in sheet.face_df[sheet.face_df["is_mesoderm"]].index:
            x = sheet.face_df.loc[f, "x"]
            c_rate = constriction_rate(
                x, max_constriction_rate=1.32, k=0.19, w=25
            )

            delam_kwargs = sheet.settings["delamination"].copy()
            delam_kwargs.update(
                {
                    "face_id": f,
                    "contract_rate": c_rate,
                    "current_traction": 0,
                    "max_traction": 30,
                }
            )
            manager.append(constriction, **delam_kwargs)

        # Progress indicator
        with progress(total=self.stop) as pbr:
            pbr.set_description("Starting simulation")

            # showing the activity dock so we can see the progress bars
            # self.viewer.window._status_bar._toggle_activity_dock(True)

            while (
                manager.current and self.t < self.stop and self.running == True
            ):
                self.t += 1

                # Clean radial tension on all vertices
                sheet.vert_df["radial_tension"] = 0
                manager.execute(sheet)
                res = solver.find_energy_min(sheet, geom, model, **solver_kw)
                self.history.record()

                manager.update()
                manager.clock += 1

                pbr.update(1)
                pbr.set_description(f"Simulation step {self.t}")
                self._on_simulation_update()

        color = sheet.vert_df["y"]

    @ensure_main_thread
    def _on_simulation_update(self):
        """
        This function is called after every simulated timestep.
        """
        LOGGER.debug("ApoptosisWidget._on_simulation_update: timestep")

        specs_kw = {}
        draw_specs = sheet_spec()
        spec_updater(draw_specs, specs_kw)
        coords = ["x", "y", "z"]

        sheet = self.history.retrieve(self.t)
        meshes = _get_meshes(sheet, coords, draw_specs)
        mesh = meshes[0]
        print(mesh)
        print(f"mesh: ({mesh[0].shape}, {mesh[1].shape}, {mesh[2].shape})")

        try:
            # if the layer exists, update the data
            self.viewer.layers["tyssue: apoptosis"].data = mesh
        except KeyError:
            # otherwise add it to the viewer
            self.viewer.add_surface(
                mesh,
                colormap="turbo",
                opacity=0.9,
                contrast_limits=[0, 1],
                name="tyssue: apoptosis",
            )
