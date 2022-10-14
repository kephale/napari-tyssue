import numpy as np

from napari_tyssue import ApoptosisWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_apoptosis_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    my_widget = ApoptosisWidget(viewer)

    # call our widget method
    my_widget._on_start_click()

    my_widget._on_stop_click()
