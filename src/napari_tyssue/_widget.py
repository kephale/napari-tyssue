"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import logging
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget

if TYPE_CHECKING:
    import napari

LOGGER = logging.getLogger("napari_tyssue.TyssueWidget")


class TyssueWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

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

    def _on_start_click(self):
        LOGGER.info("start: napari has", len(self.viewer.layers), "layers")

    def _on_stop_click(self):
        LOGGER.info("stop: napari has", len(self.viewer.layers), "layers")

    def _on_export_click(self):
        LOGGER.info("export: napari has", len(self.viewer.layers), "layers")


# This widget wraps the apoptosis demo from tyssue.
# https://github.com/DamCB/tyssue-demo/blob/master/B-Apoptosis.ipynb
class ApoptosisModelWidget(TyssueWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        # Add model parameters for config

        # Setup the UI
        self._init_buttons()

        # Add a new callback for the timeslider
