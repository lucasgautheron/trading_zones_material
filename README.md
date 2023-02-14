# How research programs come apart: code and data

This repository contains the code and data necessary to reproduce the figures and tables in "How research programs come apart: the case of supersymmetry and the disunity of physics". 

The repository is structured as follows:

| **Location**        | **Contents**                                                                                                                    |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------|
| ``inspire-harvest`` | Subdataset including the data from which present analyses are derived                                                           |
| ``analyses``        | This folder contains the scripts which generate intermediate results for the plots and tables.                                  |
| ``plots``           | This folder contains the plots included in the manuscript or supplementary materials as well as the scripts that produce them.  |
| ``tables``          | This folder contains the tables included in the manuscript or supplementary materials as well as the scripts that produce them. |
| ``output``          | This folder contains intermediate results used to produce material included in the paper.                                       |
| ``AbstractSemantics`` | This folder contains a small python package for the multithreaded retrieval of ngrams matching certain patterns from large corpora.                                       |

## Setup

### Getting the data

In order to use this repository, DataLad is required. DataLad enables reproducible science with large datasets, and instructions for its installation on various systems can be found [here](https://handbook.datalad.org/en/latest/intro/installation.html).

Once DataLad is installed on your system, we recommend creating an account on the data sharing platform [GIN](https://gin.g-node.org/), where our data are hosted, and configuring your SSH key in the parameters of your account.

The dataset can be installed by doing:

```bash
datalad install -r git@gin.g-node.org:/lucasgautheron/trading_zones_material.git
```

You can then retrieve the data with:

```bash
cd trading_zones_material
datalad get inspire-harvest/database -s s3
```

### Running analyses

Before running analyses, it is necessary to install the required python packages.
For that, please do:

```bash
pip install -r requirements.txt
```

This should cover the packages used in the present analyses. If not, please install missing dependencies manually and report them by creating a ticket.

Then, you can run the analyses you want to reproduce or reuse by doing:

```bash
python analyses/<name_of_the_analysis>.py
```

It is similarly possible to run the scripts in ``plots`` and ``tables`` for reproducing the material included in the paper and in the supplementary materials.

