## Setting up

Before you can use the `clm` package, ensure that your system meets the following prerequisites:

- Python 3.10 or later
- Pip package manager

Clone the repository and enter it:
   ```
   git clone https://github.com/skinniderlab/CLM.git
   cd CLM
   ```

## Creating a clm environment

All of `clm`'s dependencies are on [PyPI](https://pypi.org/). We have developed and tested `clm` on Python 3.10, but it should work on later Python versions as well.
You can create a new virtual environment using `venv`, and install dependencies using `pip`.

1. Verify that Python 3.10 or newer is installed on your system.
   ```
   python --version
   ```

2. Create a new environment and activate it.
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

3. In the activated environment, install all of `clm`'s dependencies.
    ```
    pip install -r requirements.txt
    ```

4. In the activated environment, install the CLM package in editable mode.
    ```
    pip install -e . --no-deps
    ```

### Using a different Python version than the system default

If your Python version is not 3.10 or later, or if you're getting errors when using a non-tested Python version, we recommend using the [uv](https://github.com/astral-sh/uv) tool to create a virtual environment with the correct Python version.
`uv` is quick to [install](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) and easy to use, both locally as well as on research clusters.

Once `uv` is installed:

1. Create a new environment with Python 3.10 and activate it.
   ```
   uv venv --python 3.10
   source .venv/bin/activate
   ```

2. In the activated environment, install all of `clm`'s dependencies.
    ```
    uv pip install -r requirements.txt
    ```

3. In the activated environment, install the CLM package in editable mode.
    ```
    uv pip install -e . --no-deps
    ```

### clm environment using conda

If you prefer using `conda`, you can use the provided `environment.yml` file to create a new conda environment with all the necessary dependencies.
> **Note**
>
> `clm` can get all its dependencies from `pypi` using `pip` and does not need [conda](https://docs.anaconda.com/miniconda/) for environment management.
Nevertheless, this might be the easiest option for most users who already have access to the `conda` executable locally or through a research cluster. The provided `environment.yml` file
has the defaults channel disabled, and can be used to create a new conda environment with all the necessary dependencies.
**It can therefore be used without getting a business or enterprise license from Anaconda. (See [Anaconda FAQs](https://www.anaconda.com/pricing/terms-of-service-faqs))**

1. Create a new conda environment named `clm` with Python version 3.10.
   ```
   conda create --name clm python=3.10 pip
   ```

2. Activate the environment.
   ```
   conda activate clm
   ```
   The command prompt will change to indicate the new conda environment by prepending `(clm)`.

3. In the activated environment, install the dependencies provided in `environment.yml`:
    ```
    conda env update --file environment.yml
    ```

4. In the activated environment, install the CLM package in editable mode.
    ```
    pip install -e . --no-deps
    ```

### Installation on specific clusters

The following instructions apply to installing and using `clm` on specific clusters at
Princeton University where we have tested out the package in the past.

#### Della
Make sure that `conda` is in your path by loading the appropriate module:

```
module load anaconda3/2024.10
```
You can also load a different anaconda version. To see all available modules, run `module avail anaconda3`.

No further actions are required to install conda or python. You can go ahead and create a new conda environment as instructed above.

#### Argo
The default `conda` is recommended on argo systems. You do not need to load any modules or do anything special to use the conda command on argo. [Click here for detailed information regarding conda on Argo](https://lsidocs.princeton.edu/index.php/Conda).
