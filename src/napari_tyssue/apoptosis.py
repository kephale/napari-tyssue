"""
This module implements a plugin that supports the tyssue apoptosis demo.

This module was derived from https://github.com/DamCB/tyssue-demo, a MPLv2
licensed project.
"""
import logging
import sys
import pooch

import napari

import numpy as np

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

LOGGER = logging.getLogger("napari_tyssue.ApoptosisWidget")

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)

from napari_tyssue.tyssuewidget import TyssueWidget, _get_meshes


# This widget wraps the apoptosis demo from tyssue.
# https://github.com/DamCB/tyssue-demo/blob/master/B-Apoptosis.ipynb
class ApoptosisWidget(TyssueWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

        # tyssue model init

        # The history stores simulation outputs
        self.history = None

        # Current timestep
        self.t = 0

        # This is the stop time of the simulation
        self.stop = 100

        # Add model parameters for config

        # Setup the UI
        self._init_buttons()

        # Add a new callback for the timeslider

    def start_simulation(self):
        # Read pre-recorded datasets

        h5store = pooch.retrieve(
            url="https://github.com/DamCB/tyssue-demo/raw/master/data/small_hexagonal.hf5",
            known_hash=None,
            progressbar=True,
        )

        datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])

        # Corresponding specifications
        specs = config.geometry.cylindrical_sheet()
        sheet = Sheet("emin", datasets, specs)
        sheet.sanitize(trim_borders=True, order_edges=True)

        geom.update_all(sheet)

        # Model
        nondim_specs = config.dynamics.quasistatic_sheet_spec()
        dim_model_specs = model.dimensionalize(nondim_specs)
        sheet.update_specs(dim_model_specs)

        sheet.get_opposite()
        live_edges = sheet.edge_df[sheet.edge_df["opposite"] == -1].index
        dead_src = sheet.edge_df.loc[live_edges, "srce"].unique()

        ### Boundary conditions
        sheet.vert_df.is_active = 1
        sheet.vert_df.loc[dead_src, "is_active"] = 0

        sheet.edge_df["is_active"] = sheet.upcast_srce(
            "is_active"
        ) * sheet.upcast_trgt("is_active")

        # Energy minimization

        min_settings = {
            #    "minimize":{
            "options": {"disp": False, "ftol": 1e-6, "gtol": 1e-5},
            #    }
        }
        solver = QSSolver()

        res = solver.find_energy_min(sheet, geom, model, **min_settings)
        LOGGER.info((res["success"]))

        # Choose apoptotic cell
        # TODO this cell selection could be interactive

        apoptotic_cell = 16
        LOGGER.info(
            "Apoptotic cell position:\n{}".format(
                sheet.face_df.loc[apoptotic_cell, sheet.coords]
            )
        )
        apoptotic_edges = sheet.edge_df[
            sheet.edge_df["face"] == apoptotic_cell
        ]
        apoptotic_verts = apoptotic_edges["srce"].values
        LOGGER.info(
            "Indices of the apoptotic vertices: {}".format(apoptotic_verts)
        )

        from tyssue.behaviors.sheet import apoptosis
        from tyssue.behaviors import EventManager

        manager = EventManager("face")

        sheet.settings["apoptosis"] = {
            "shrink_rate": 1.2,
            "critical_area": 8.0,
            "radial_tension": 0.2,
            "contractile_increase": 0.3,
            "contract_span": 2,
        }

        sheet.face_df["id"] = sheet.face_df.index.values
        manager.append(
            apoptosis, face_id=apoptotic_cell, **sheet.settings["apoptosis"]
        )

        # Run the simulation
        self.t = 0

        self.history = History(sheet)
        self.running = True

        self.viewer.dims.ndisplay = 3

        # Progress indicator
        with progress(total=self.stop) as pbr:
            pbr.set_description("Starting simulation")

            # showing the activity dock so we can see the progress bars
            # self.viewer.window._status_bar._toggle_activity_dock(True)

            while (
                manager.current and self.t < self.stop and self.running == True
            ):
                manager.execute(sheet)                
                res = solver.find_energy_min(
                    sheet, geom, model, **min_settings
                )
                self.history.record()
                manager.update()

                pbr.update(1)
                pbr.set_description(f"Simulation step {self.t}")
                self._on_simulation_update()
                self.t += 1

        color = sheet.vert_df["y"]

    @ensure_main_thread
    def _on_simulation_update(self):
        """
        This function is called after every simulated timestep.
        """
        LOGGER.debug("ApoptosisWidget._on_simulation_update: timestep")

        specs_kw = {}
        draw_specs = sheet_spec()
        draw_specs["vert"]["visible"] = True
        spec_updater(draw_specs, specs_kw)
        coords = ["x", "y", "z"]

        layer_name = "tyssue: apoptosis"

        vert_offset = 0
        if layer_name in self.viewer.layers:
            # Offset is # of vertices for all timepoints
            vert_offset = self.viewer.layers[layer_name].data[0].shape[0]

        sheet = self.history.retrieve(self.t)
        meshes = _get_meshes(sheet, coords, draw_specs)
        vertices, faces, values = meshes[0]

        LOGGER.info(
            f"num_meshes = {len(meshes)} mesh: ({vertices.shape}, {faces.shape}, {values.shape})"
        )

        num_verts = vertices.shape[0]

        # Now we need to make the mesh into a timepoint
        tp_vertices = np.concatenate(
            (np.ones((num_verts, 1)) * self.t, vertices), axis=1
        )
        tp_faces = faces + vert_offset

        # rho = np.linalg.norm(self.monolayer.vert_df[self.monolayer.coords], axis=1)
        
        # Use rho value for coloring
        # values = np.asarray(sheet.vert_df["rho"])

        # multichannel values test
        #values = np.concatenate((values.reshape((1, num_verts)), values.reshape((1, num_verts))), axis=0)

        try:
            # if the layer exists, update the data
            curr_verts, curr_faces, curr_values = self.viewer.layers[
                layer_name
            ].data

            self.viewer.layers[layer_name].data = (
                np.concatenate((curr_verts, tp_vertices), axis=0),
                np.concatenate((curr_faces, tp_faces), axis=0),
                # For multichannel
                # np.concatenate((curr_values, values), axis=1),
                np.concatenate((curr_values, values), axis=0),
            )

            # Update timepoint that is displayed
            self.viewer.dims.set_current_step(0, self.t)
            
        except KeyError:
            # otherwise add it to the viewer
            self.viewer.add_surface(
                (tp_vertices, tp_faces, values),
                colormap="viridis",
                opacity=0.9,
                contrast_limits=[0, 1],
                name=layer_name,
            )


if __name__ == "__main__":
    viewer = napari.Viewer()

    # LOGGER.setLevel(logging.DEBUG)
    widget = ApoptosisWidget(viewer)

    viewer.window.add_dock_widget(widget, name="napari-tyssue")
    
    # widget.start_simulation()

