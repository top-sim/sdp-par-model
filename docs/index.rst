.. SKA SDP Parametric Model documentation master file, created by
   sphinx-quickstart on Mon Jan 23 13:58:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. py:currentmodule:: sdp_par_model

SKA SDP Parametric Model
************************

The SDP parametric model is a collection of equations and software
code that model the performance aspects of the SDP. We created an
interactive software interface for performing computations and
displaying their results so that people within SDP can easily interact
and experiment with the model. Python, IPython and the Jupyter
Notebook are the tools that we used to accomplish this.

Interactive functionality is implemented in IPython, a command shell
for interactive computing. Although IPython currently supports an
extensive set of programming languages, Python is the only programming
language used in the SDP Parametric model.

Jupyter Notebook (formerly known as IPython Notebook) is a web
application that allows you to create “notebook” documents that
contain live code, equations, visualisations and explanatory
text. Being a web application, Jupyter runs in your web browser while
in the background interfacing with a “kernel”. When you open a
notebook on your own computer a kernel will automatically start up and
run locally. Your web browser will open a new tab that interfaces to
this local kernel (without requiring an internet connection).

(IPython) Jupyter notebook sessions may be saved as files with the
.ipynb extension. A notebook file can be shared with other people in
much the same way as MS Word or PDF document. A notebook including all
formatted text, output and plotted figures can be exported to a static
PDF file. Alternatively, a notebook may be used to interactively
re-compute results (when used in conjunction with the accompanying
code).

For an overview of The IPython project, see: http://ipython.org/

For an overview of Project Jupyter, see: http://jupyter.org/

The “SDP parametric model” contains several Jupyter Notebook
files. These will be briefly listed and explained later in this
document. In the rest of this document either “IPython” and/or
“notebook” will refer to our Python and IPython Jupyter Notebook
implementation as a whole.

A central goal of the notebook is to compute performance metrics and
general sizing estimates for the SDP. We attempted a 1:1
implementation of the equations described in AD01. As the notebook is
being developed in parallel with AD01, some discrepancies may
occur. Refer to AD01 for the fundamental choices in which algorithms
are modelled. The w-snapshot algorithm with the use of faceting is
assumed. Certain parameters may be toggled, e.g. where uncertainty
remains (e.g. in choices relating to baseline-dependent convolution
kernel sizes and baseline-dependent gridding coalescing).

Version Control
===============

The IPython Notebook is hosted on Github as part of the following
“SDP-par-model” project. You’ll need to be a member of the `SKA Github
group <https://confluence.ska-sdp.org/display/SP/GitHub>`_ in order to
access this repository:

https://github.com/SKA-ScienceDataProcessor/sdp-par-model

If you’re not familiar with using git, a user-friendly GUI such as
`Atlassian Sourcetree
<https://www.atlassian.com/software/sourcetree/overview>`_ (available
for Mac & Windows) is a way to get started. A list of other git GUIs
can be found `here <http://git-scm.com/download/gui/linux>`_.

The development of the IPython model has so far been carried out by
many sometimes-overlapping contributions. Direct contributors to the
code repository include Rosie Bolton, Anna Scaife, Bojan Nikolic,
Juande Santander-Vela, Francois Malan, Peter Wortmann and Tim
Cornwell. These contributions are often committed to different
branches that are later merged. The most up-to-date working copy of
the model is typically represented by the head of the “master” branch.

The evolution of the IPython model is recorded in git as a series of
immutable revisions that is each uniquely identified by a
40-character ID. We may refer to a revision by the first 7 characters
of its ID, e.g. [9ffbb2e]. Descriptive git “tags” are additionally
used to identify major milestones. Because the IPython model is
continually evolving, each document that uses the model’s results
(e.g. RD03) should reference the unique revision ID that was used for
its generation.

SDP notebooks
=============

The parametric model has several notebook files in the ``/iPython``
sub-folder that illustrate different parts of the Model. Ranked in
order of typical importance (high to low) these are:

SKA1_Imaging_Performance_Model
------------------------------

This notebook contains cells for computing tables of values (data
rates, FLOPs, image sizes etc), as well as for generating 1D and 2D
parameter sweep plots.

SKA1_Export
-----------

Allows exporting detailed parametric model output for usage with
external programs. This notebook can also be used to compare existing
parametric model exports to allow fine-grained change tracking on
computation results.

SKA1_SDP_DesignEquations
------------------------

This algorithmically constructs the “design equations” contained in
AD01 by using the IPython model’s underlying code. Symbolic equations
are formatted in human-readable format, while numerical limits
corresponding to these equations are also computed.

SKA1_Dataflow
-------------

This creates a graphical (graph) representation of the SDP data
pipeline. It uses the IPython model to illustrate numerical data
rates, computational loads, and various other parameters in the
output. The graph can be exported as a PDF files or PNG images.

SKA1_Faceting_Model
------------------------------

This investigates the influence of one implementation of faceting in
the SKA Performance Model. It generates a host of values and 1D and 2D
plots. This notebook provides the plots for the TCC memo
``TCC-SDP-151123-1-1``.

SKA1_Sensitivity_Analysis
------------------------------

This investigates the parameter sensitivity in the SKA Performance
Model.

Absolute_Baseline_length_distribution
-------------------------------------

This notebook reads antenna station locations (as latitude / longitude
coordinates) from text file (located in the ``/data/layouts``
subdirectory) to compute and visualize the baseline
distributions. This notebook is determining the baseline bins that go
into the parametric model calculations.

SKA1_SDP_Products
-------------------------------------

Generates all product cost formulas for all pipelines. Useful for
checking what exactly the parametric model calculates.

Code documentation
==================

Python code can be executed directly in a notebook and is indeed
entered and executed directly in some of the notebooks listed in the
previous section. It is, however, best practice to keep the bulk of
the code separate from the notebooks themselves. This practice
facilitates easier code writing, better integration with integrated
development environments (such as PyCharm), better coding practice,
easier source control, and more advanced features (such as unit
testing).

Code that is maintained in proper Python code libraries (with .py
extension) can be imported into a notebook using standard Python
syntax. This is typically done in the first code block of a notebook
file. Two important concepts are those of a
:class:`.config.PipelineConfig`, which represents a pipeline
configuration to query properties of, and
:class:`.parameters.container.ParameterContainer`, which is used to
accumulate information we have about such a telescope configuration.

Below follows the general structure of the IPython Model’s code base,
ranked in order of decreasing importance:

.. toctree::
   toplevel
   parameters
   dataflow
   :maxdepth: 2


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

