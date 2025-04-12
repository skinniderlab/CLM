.. CLM documentation master file, created by
   sphinx-quickstart on Thu Apr 18 03:52:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CLM's documentation!
===============================

A package to train and evaluate language models of chemical structures,
as used in the manuscript, :cite:t:`Qiang2024`.

Note that training and evaluating chemical language models requires significant
computational resources. Running the default `config.yaml` file distributed
with the repository, for example, will submit a total of 2102 jobs, with an
average requested runtime of ~35 hours per job and a maximum memory of 256 GB.
Actual runtimes will generally be substantially below this (but, for unusually
large datasets, may be longer and require modification of the default resource
requests), and resource requirements will vary substantially as a function of
the configuration, with key parameters including the dataset size, number of
cross-validation folds, degree of non-canonical SMILES enumeration, and amount
of sampling to perform from the trained models (for instance, performing
three-fold cross-validation and non-canonical SMILES enumeration at a fixed
augmentation factor of 10x would reduce this resource request by approximately
20-fold). Nonetheless, access to a high-performance computing cluster with GPU
resources is strongly recommended to train and evaluate new models.

If you are simply looking to use the trained DeepMet model to annotate
metabolomics data, consider using the
DeepMet `web application <https://deepmet.org/>`_.

.. toctree::
   :titlesonly:

   clm/installation.md
   clm/workflow.md
   clm/workflow_steps.md

.. bibliography::
