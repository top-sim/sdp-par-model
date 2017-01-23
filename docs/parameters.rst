
Parameter Definitions
=====================

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
