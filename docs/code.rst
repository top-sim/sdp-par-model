
Implementation
==============

Most of the parametric code is maintained as Python code libraries,
which is imported by the notebooks. Two important concepts are those
of a :class:`.config.PipelineConfig`, which represents a pipeline
configuration to query properties of, and
:class:`.parameters.container.ParameterContainer`, which is used to
accumulate information we have about such a telescope configuration.

Below follows the general structure of the model's code base:


config
------

.. automodule:: sdp_par_model.config
   :members:
   :undoc-members:

evaluate
--------

.. automodule:: sdp_par_model.evaluate
   :members:
   :undoc-members:

reports
-------

.. automodule:: sdp_par_model.reports
   :members:
   :undoc-members:

.. py:currentmodule:: sdp_par_model.parameters

As the heart of the parametric model, these modules generate
parametric equations from parameter definitions. The whole process is
very open and flexible: Virtually all methods work by updating a
common :class:`.container.ParameterContainer`
object step by step until all required equations have been
applied.

Note that this is designed to work no matter whether the parameters
are given symbolically (as sympy expressions) or as plain values. Thus
this code can be used to generate both values or formulae, with
fine-grained control over terms at every calculation step.

The general workflow is that a fresh parameter container will get
populated using data from :mod:`.definitions` appropriate for the
desired telescope configuration. Then :mod:`.equations` will be used
to derive parameter equations from them. The calculation can be
"symbol-ified" at multiple stages, for example using
:meth:`.definitions.define_design_equations_variables` or
:meth:`.equations.apply_imaging_equations`.

parameters.container
--------------------

.. automodule:: sdp_par_model.parameters.container
   :members:
   :undoc-members:

parameters.definitions
----------------------

.. automodule:: sdp_par_model.parameters.definitions
   :members:
   :undoc-members:

parameters.equations
--------------------

.. automodule:: sdp_par_model.parameters.equations
   :members:
   :private-members:
   :undoc-members:
