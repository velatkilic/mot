# XPCI Multi Object Tracker

A library for automatic quantitative characterization of X-ray Phase Contrasting Imaging videos of combustions.

## Installation
We recommend using the library in a Unix/Linux system. For Windows users, the `Windowns Subsystem for Linux (WSL)` can be easily installed following the instructions [here](https://learn.microsoft.com/en-us/windows/wsl/install).

1. Install `git` if you don't already have one. For example, on Ubuntu,

    `sudo apt-get install git`

2. Download the `xmot` source code.

    `git clone https://github.com/velatkilic/mot.git`

3. Install and upgrade the necessary build packages of python.

    `python3 -m pip install --user --upgrade pip build`

4. Compile and install `xmot`. Dependent packages are listed in the file `requirements.txt`. They will be downloaded automatically during installation.

    * [Recommanded] For developers and users want always keep their code updated to the latest version, include `-e` of `pip` to enable the development mode. Updates of the library will be automatically built and reflected in usage. 


        ```
        cd mot
        python3 -m pip install --user -e .
        ```

    * For regular users who just want to install a static version of the library

        ```
        cd mot
        python3 -m pip install --user .
        ```

5. Check whether the installation has succeeded. If the installation succeeded without a problem, users should be able to import `xmot` without errors.

    ```
    python

    # In the python prompt
    > import xmot
    ```

## Usage
The process of analyzing a video can be largely separated into two steps:
1.  Detecting particles from each frame of the video; 
2.  Analyzing detected particles and extract information;

We have included example scripts for the two steps in the folder `examples/video_1`: `particle_detection.py` and `particle_analysis.py`.