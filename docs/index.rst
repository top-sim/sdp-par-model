
.. py:currentmodule:: sdp_par_model

SKA SDP Parametric Model
************************

The SDP parametric model is a collection of equations and software
code that model the performance aspects of the SDP. We created an
interactive software interface for performing computations and
displaying their results so that people both within and external to
SDP can easily interact and experiment with the model. This is
accomplused using Python and Jupyter notebooks.  The parametric model
contains several Jupyter Notebook files. These will be briefly listed
and explained later.

A central goal of the notebooks is to compute performance metrics and
general sizing estimates for the SDP. Refer to the parametric model
document for the fundamental choices of which algorithms are
modelled. The basic assumption is that we employ a combination of
w-snapshots and w-stacking combined with facetting for distribution.

SDP notebooks
=============

The parametric model has several notebook files in the ``/iPython``
sub-folder that illustrate different parts of the Model. Ranked in
order of typical importance (high to low) these are:

`SKA1_Imaging_Performance_Model
<iPython/SKA1_Imaging_Performance_Model.html>`_:
Contains cells for computing tables of values (data
rates, FLOPs, image sizes etc), as well as for generating 1D and 2D
parameter sweep plots.

`SKA1_Export
<iPython/SKA1_Export.html>`_:
Allows exporting detailed parametric model output for usage with
external programs. This notebook can also be used to compare existing
parametric model exports to allow fine-grained change tracking on
computation results.

`SKA1_SDP_DesignEquations
<iPython/SKA1_SDP_DesignEquations.html>`_:
This algorithmically constructs the “design equations” contained in
AD01 by using the IPython model’s underlying code. Symbolic equations
are formatted in human-readable format, while numerical limits
corresponding to these equations are also computed.

`SKA1_Dataflow
<iPython/SKA1_Dataflow.html>`_:
This creates a graphical (graph) representation of the SDP data
pipeline. It uses the IPython model to illustrate numerical data
rates, computational loads, and various other parameters in the
output. The graph can be exported as a PDF files or PNG images.

`SKA1_Faceting_Model
<iPython/SKA1_Faceting_Model.html>`_:
This investigates the influence of one implementation of faceting in
the SKA Performance Model. It generates a host of values and 1D and 2D
plots. This notebook provides the plots for the TCC memo
``TCC-SDP-151123-1-1``.

`SKA1_Sensitivity_Analysis
<iPython/SKA1_Sensitivity_Analysis.html>`_:
This investigates the parameter sensitivity in the SKA Performance
Model.

`Absolute_Baseline_length_distribution
<iPython/Absolute_Baseline_length_distribution.html>`_:
This notebook reads antenna station locations (as latitude / longitude
coordinates) from text file (located in the ``/data/layouts``
subdirectory) to compute and visualize the baseline
distributions. This notebook is determining the baseline bins that go
into the parametric model calculations.

`SKA1_SDP_Products
<iPython/SKA1_SDP_Products.html>`_:
Generates all product cost formulas for all pipelines. Useful for
checking what exactly the parametric model calculates.

Code documentation
==================

Most of the parametric code is maintained as Python code libraries,
which is important by the notebooks. Two important concepts are those
of a :class:`.config.PipelineConfig`, which represents a pipeline
configuration to query properties of, and
:class:`.parameters.container.ParameterContainer`, which is used to
accumulate information we have about such a telescope configuration.

Below follows the general structure of the model's code base:

.. toctree::
   toplevel
   parameters
   :maxdepth: 2

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

