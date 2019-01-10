scheduling
==========

The scheduling module contains functions to generate a sequence of
observations from a list of projects (HPSOs).

There are two principal functions:

extract_projects
----------------

Extracts projects and their relevant parameters from the parametric
model output.

projects = extract_projects(input_file, telescope_name)

    input_file: name of CSV file containing parametric model output
telescope_name: name of telescope ('SKA1_Low' or 'SKA1_Mid')
      projects: list of projects (numpy structured array)

The list of projects has fields:

- name     name of project (HPSO-x)
- tpoint   time to observe a single pointing [s]
- texp     total observation time (length of project) [s]
- r_rflop  real-time compute rate [PFLOP/s]
- b_rflop  batch compute rate [PFLOP/s]
- rinp     input visibility data rate [TB/s]
- rout     output data rate for data produced continuously (averaged visibilities) [TB/s]
- mout     output data size for data produced per pointing (images) [TB]

generate_sequence
-----------------

Generate random sequence of observations given a list of projects.

sequence = generate_sequence(projects, tsched, tseq, allow_short_tobs)

        projects: list of projects (numpy structured array)
          tsched: maximum length of scheduling block [s]
            tseq: length of sequence to generate [s]
allow_short_tobs: allow observations with short duration (default is False)
        sequence: list of observations (numpy structured array)

If allow_short_tobs is False, then all of the observations will be of
length tsched. If it is true, then projects with tpoint less than
tsched will have tobs set to tpoint, otherwise tobs will be set to
tsched.

The list of observations has fields:

- uid      unique ID (set to the index in the sequence)
- name     name of project
- tobs     observation time [s]
- r_rflop  real-time compute rate [PFLOP/s]
- b_rflop  batch compute rate [PFLOP/s]
- minp     input data size (visibilities) [TB]
- mout     output data size (images and averaged visibilities) [TB]
