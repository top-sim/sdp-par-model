
Notebooks
*********

Notebooks are the main user interface of the SKA SDP parametric
model. To allow exploration of the design space, the parametric model
provides a number of separate notebooks for different purposes.

Build Instructions
==================

If you wish to use the notebooks interactively, clone the parametric
model repository and run the notebooks yourself::

  $ git clone https://github.com/ska-telescope/sdp-par-model
  $ cd sdp-par-model
  $ pipenv install
  $ pipenv shell
  $ jupyter notebook notebooks

This should allow you to interact with the notebooks using a web browser.

List of Notebooks
=================

The following notebooks come with the parametric model. Follow the
links for non-interactive version of the notebooks:

`SKA1_Imaging_Performance_Model
<http://ska-telescope.gitlab.io/sdp-par-model/notebooks/SKA1_Imaging_Performance_Model.html>`_:
Contains interactive cells for computing telescope parameters (data
rates, FLOPs, image sizes etc).

`SKA1_Sensitivity_Analysis
<http://ska-telescope.gitlab.io/sdp-par-model/notebooks/SKA1_Sensitivity_Analysis.html>`_:
Investigates the sensitivity of telescope parameters on inputs such as
baseline length and ionospheric time scale.

`SKA1_SDP_Performance_Dashboard
<http://ska-telescope.gitlab.io/sdp-par-model/notebooks/SKA1_SDP_Performance_Dashboard.html>`_:
Overview of FLOP rate predictions for different telescopes, pipelines
and high-priority science objectives (HPSOs).

`SKA1_Export
<http://ska-telescope.gitlab.io/sdp-par-model/notebooks/SKA1_Export.html>`_:
Allows exporting detailed parametric model output for usage with
external programs. This notebook can also be used to compare existing
parametric model exports to allow fine-grained change tracking of
calculated parameters.

`SKA1_System_Sizing
<http://ska-telescope.gitlab.io/sdp-par-model/notebooks/SKA1_System_Sizing.html>`_:
Uses the HPSOs to estimate the average computational requirements and
output data rates of the SDP instances at the two telescopes. This
informs the system sizing.

`SKA1_SDP_Products
<http://ska-telescope.gitlab.io/sdp-par-model/notebooks/SKA1_SDP_Products.html>`_,
`SKA1_Document_Formulas
<http://ska-telescope.gitlab.io/sdp-par-model/notebooks/SKA1_Document_Formulas.html>`_:
Shows the formulas used by the parametric model for calculating
parameters. Useful for checking what exactly the parametric model
calculates.

`Absolute_Baseline_length_distribution
<http://ska-telescope.gitlab.io/sdp-par-model/notebooks/Absolute_Baseline_length_distribution.html>`_:
Compute and visualize baseline distributions of SKA telescopes. This
notebook is used for determining the baseline bins that go into the
parametric model calculations.
