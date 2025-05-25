<a id="readme-top"></a>


<br />
<div align="center">
  <a href="https://gitlab.com/cmbm-ethz/bamboost">
	<img src="https://gitlab.com/cmbm-ethz/bamboost/-/raw/main/assets/bamboost_icon.png?ref_type=heads" width="150" alt="Logo"/><br/>
  </a>

<h3 align="center">BAMBOOST</h3>

  <p align="center">
    Bamboost is a Python library built for datamanagement using
    the HDF5 file format.
    bamboost stands for a <span style="font-weight: bold;">lightweight</span> shelf which will <span style="font-weight: bold">boost</span> your efficiency and which
    will totally break if you load it heavily. Just kidding, bamboo can fully carry pandas. <br/>
    🐼🐼🐼🐼
    <br />

[![Docs][docs-shield]][docs-url]
[![Pipeline][pipeline-shield]][pipeline-url]
[![Coverage][coverage-shield]][coverage-url]
[![Pypi][pypi-shield]][pypi-url]
[![PyPI_downloads][pypi-downloads-shield]][pypi-downloads-url]
[![License][license-shield]][license-url]


<a href="https://bamboost.ch/docs"><strong>Explore the docs »</strong></a>
<br />
<br />
<a href="https://github.com/zrlf/bamboost-tui">Terminal User Interface (TUI)</a>
&middot;
<a href="https://github.com/zrlf/bamboost-docs">Doc site repo</a>
<br />
<br />
<a href="https://gitlab.com/cmbm-ethz/bamboost/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
&middot;
<a href="https://gitlab.com/cmbm-ethz/bamboost/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>

  </p>
</div>

> [!important]
> Starting from version 0.10.0, bamboost breaks compatibility with previous versions.
> For previous versions, checkout the [legacy branch](https://gitlab.com/cmbm-ethz/bamboost/-/tree/legacy).

<details>
  <summary>Table of Contents</summary>

[[_TOC_]]

</details>



## About The Project

**bamboost** is a python data framework designed for managing scientific simulation data.
It provides an organized model for storing, indexing, and retrieving
simulation results, making it easier to work with large-scale computational
studies. 
In its core, it is a filesystem storage model, providing directories for
**simulations**, bundled in **collections**. 

### Principles
- **Independence:** Any dataset must be complete and understandable on it's own. You can copy or extract any of your data and distribute it without external dependencies.
- **Path redundancy:** Data must be referencable without knowledge of it's path. This serves several purposes: You can share your data easily ($e.g.$ supplementary material for papers), and renaming directories, moving files, switching computer, etc. will not break your data referencing.

This leads to the following requirements:
- Simulation parameters must be stored locally, inside the simulation directory. Crucially, not _exclusively_ in a global database of any kind.
- Collections must have unique identifiers that are independent of its path.
- Simulations must have unique identifiers that are independent of its path.

### Concept

We organize **simulations** in **collections** within structured
directories. 
Let's consider the following directory:

```
test_data/
├── simulation_1/
│   ├── data.h5
│   ├── data.xdmf
│   ├── additional_file_1.txt
│   ├── additional_file_2.csv
├── simulation_2/
│   ├── data.h5
│   ├── additional_file_3.txt
└── .bamboost-collection-ABCD1234
```

This is a valid `bamboost` collection at the path `./test_data`. It contains an
identifier file giving this collection a unique identifier. In this case, it is
`ABCD1234`.
This file defines the unique ID of the collection.

It contains two entries; `simulation_1` and `simulation_2`.
As you can see, each simulation owns a directory inside a collection. 
The directory names are simultaneously used as their _name_ as well as their _ID_.
The unique identifier for a single simulation becomes the combination of the collection _ID_ that it belongs to and the simulation _ID_.
That means, the full identifier of `simulation_1` is `ABCD1234:simulation_1`.

Each simulation contains a central _HDF5 file_ named `data.h5`. This file is used to store the _parameters_, as well as generated data.
The simulation API of `bamboost` provides extensive functionality to store and retrieve data from this file. However, users are not limited to this file, or using `python` in general.
The reason why simulations are directories instead of just a single HDF file is that you can dump any file that belongs to this simulation into its path. This can be output from 3rd party software (think LAMMPS), additional input files such as images, and also scripts to reproduce the generated data.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting Started

**bamboost** is available from the Python Package Index (PyPI) and can be installed using `pip` (or `uv` of course):

```sh
pip install bamboost
```

### Prerequisites

To use **bamboost** with MPI, you need a working MPI installation. Additionally, you need
- `mpi4py`
- `h5py` with MPI support. For most distros you can get a version from it's package manager, or see [building against Parallel HDF5](https://docs.h5py.org/en/stable/mpi.html#building-against-parallel-hdf5)

### Installation

**bamboost** is available from the Python Package Index (PyPI) and can be installed using `pip` (or `uv` of course):

```sh
pip install bamboost
```

To install the latest version from this repository, you can use:

```sh
pip install git+https://gitlab.com/cmbm-ethz/bamboost.git
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Usage

For a getting started guide, please see here: [Getting started](https://bamboost.ch/docs/getting_started)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Clear MPI handling

See the [open issues](https://gitlab.com/cmbm-ethz/bamboost/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the **MIT license**. See [LICENSE](https://gitlab.com/cmbm-ethz/bamboost/-/blob/main/LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

zrlf - forez@ethz.ch

Project Link: [https://gitlab.com/cmbm-ethz/bamboost](https://gitlab.com/cmbm-ethz/bamboost)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Best-README-Template](https://github.com/othneildrew/Best-README-Template) for inspiration on how to structure this README.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[docs-shield]: https://img.shields.io/badge/Docs-bamboost.ch-blue?style=flat-square
[docs-url]: https://bamboost.ch
[pipeline-shield]: https://gitlab.com/cmbm-ethz/bamboost/badges/main/pipeline.svg?style=flat-square
[pipeline-url]: https://gitlab.com/cmbm-ethz/bamboost/-/commits/main
[coverage-shield]: https://gitlab.com/cmbm-ethz/bamboost/badges/main/coverage.svg?style=flat-square
[coverage-url]: https://cmbm-ethz.gitlab.io/bamboost/
[pypi-shield]: https://img.shields.io/pypi/v/bamboost?style=flat-square
[pypi-url]: https://pypi.org/project/bamboost/
[pypi-downloads-shield]: https://img.shields.io/pypi/dm/bamboost?style=flat-square
[pypi-downloads-url]: https://pypi.org/project/bamboost/
[license-shield]: https://img.shields.io/pypi/l/bamboost?style=flat-square
[license-url]: https://gitlab.com/cmbm-ethz/bamboost/-/blob/main/LICENSE


