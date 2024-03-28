# Cuda-Beagle-Openmpi-Mrbayes
How to set-up Cuda, Beagle, Openmpi, and MrBayes

# Overview
MrBayes is a software widely used for Bayesian inference methods, especially in the fields of bioinformatics and molecular phylogenetics. It is primarily used for the Bayesian inference of phylogenetic trees, offering a statistical method to estimate the evolutionary history of sample data. MrBayes utilizes Markov Chain Monte Carlo (MCMC) methods to estimate the posterior distribution, thereby enabling researchers to infer the relationships between species and their phylogenetic trees based on genetic data.

Beagle integration
By integrating Beagle-lib, MrBayes can take advantage of the powerful computing power of GPUs and multi-core CPUs to accelerate computationally intensive Bayesian analysis. The acceleration effect of Beagle-lib can significantly reduce the time required for model inference and parameter estimation, making it possible to run high-complexity models.

OpenMPI integration
The integration of OpenMPI further enhances the parallel computing capabilities of MrBayes, allowing computing tasks to be distributed across multiple computing nodes. This approach can effectively utilize distributed computing resources and further speed up the analysis process, especially when running on large-scale clusters or supercomputers.

Mrbayes provides a version equipped with Beagle and Openmpi, which is designed to allow us to perform tree inference faster and improve the efficiency of building Bayesian phylogenetic trees. However, the Windows version of mrbayes is not equipped with Beagle and Openmpi. Installing Mrbayes equipped with Beagle and Openmpi on a Linux system is a frustrating process.

Here I will explain how to set up these programs (Cuda, Beagle, Openmpi, Mrbayes) correctly. Minimize your installation frustration. This time I will demonstrate my piggyback installation process on Ubuntu system, which has not been fully verified on other Linux distributions.

## 1 Get started
- Computer hardware parameters installed for this test:

| CPU                                                     | GPU                                                       | Memory                           |
| :-----------------------------------------------------: | :-------------------------------------------------------: | :------------------------------: |
| <font color=#813C85>Inter(R) Core(TM) i7-14650HX</font> | <font color=#813C85>NVIDIA GeForce RTX 4060 Laptop</font> | <font color=#813C85>32 GB</font> |

- Software that needs to be installed  <font color=#ED3321>(Note: NVIDIA Driver and CUDA Toolkit determine the downloaded version based on your own personal PC)</font>

|             |             Nvidia Driver             |           Cuda Toolkit            |              Beagle              |             Openmpi              | Mrbayes                          |
| :---------: | :-----------------------------------: | :-------------------------------: | :------------------------------: | :------------------------------: | :------------------------------: |
| **Version** | <font color=#813C85>535.161.07</font> | <font color=#813C85>11.8.0</font> | <font color=#813C85>3.1.2</font> | <font color=#813C85>4.1.5</font> | <font color=#813C85>3.2.7</font> |

- Operating system: <font color=#813C85>Ubuntu 22.04.4 LTS</font>

- Installation package:
1. [beagle-lib-3.1.2](https://github.com/beagle-dev/beagle-lib/archive/refs/tags/v3.1.2.tar.gz)
2. [openmpi-4.1.5](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz)
3. [mrbayes-3.2.7](https://github.com/NBISweden/MrBayes/releases/download/v3.2.7/mrbayes-3.2.7.tar.gz)
4. [Cuda-Releases](https://developer.nvidia.com/cuda-toolkit-archive)

- Other software
```bash
ubuntu-drivers devices
```

```bash
sudo ubuntu-drivers autoinstall
```
This time I installed it as NVIDIA driver metapackage from nvidia-driver-535 (proprietary).  <font color=#ED3321>(Note: Try to choose the proprietary version when installing)</font>.

After installation, restart the computer:
```bash
reboot
```

After restarting the computer, check the detailed information of the GPU to verify whether the Nvidia driver is successfully installed and the highest supported Cuda version.
```bash
nvidia-smi
```

The following is my output information. According to the information, I can know that the highest supported Cuda version is <font color=#ED3321>12.2</font>.
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    Off | 00000000:01:00.0  On |                  N/A |
| N/A   40C    P8               3W /  55W |    242MiB /  8188MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1738      G   /usr/lib/xorg/Xorg                          158MiB |
|    0   N/A  N/A      1953      G   /usr/bin/gnome-shell                         41MiB |
|    0   N/A  N/A      3016      G   ...erProcess --variations-seed-version       35MiB |
+---------------------------------------------------------------------------------------+
```

### 2.2 Install  Cuda Toolkit
Before installing Cuda, you need to determine the NVIDIA architectures, CUDA arch and CUDA gencode of your personal computer.
- [Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

| Fermi                            | Kepler                           | Maxwell                          | Pascal                           | Volta                            | Turing                           | Ampere                           | **Ada**                          | Hopper                            | Blackwell                    |
| :------------------------------: | :------------------------------: | :------------------------------: | :------------------------------: | :------------------------------: | :------------------------------: | :------------------------------: | :------------------------------: | :-------------------------------: | :--------------------------: |
| <font color=#813C85>sm_20</font> | <font color=#813C85>sm_30</font> | <font color=#813C85>sm_50</font> | <font color=#813C85>sm_60</font> | <font color=#813C85>sm_70</font> | <font color=#813C85>sm_75</font> | <font color=#813C85>sm_80</font> | <font color=#813C85>sm_89</font> | <font color=#813C85>sm_90</font>  | <font color=#813C85>?</font> |
|                                  | <font color=#813C85>sm_35</font> | <font color=#813C85>sm_52</font> | <font color=#813C85>sm_61</font> | <font color=#813C85>sm_72</font> |                                  | <font color=#813C85>sm_86</font> |                                  | <font color=#813C85>sm_90a</font> |                              |
|                                  | <font color=#813C85>sm_37</font> | <font color=#813C85>sm_53</font> | <font color=#813C85>sm_62</font> |                                  |                                  | <font color=#813C85>sm_87</font> |                                  |                                   |                              |

According to [Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/), the NVIDIA architectures of RTX 4060 are Ada, corresponding to download Cuda 11.08, So this time download and install Cuda version 11.08.

Download Cuda 11.08 according to official [Cuda-Releases](https://developer.nvidia.com/cuda-toolkit-archive). Select Linux -> x86_64 -> Ubuntu -> 22.04 -> runfile (local).
- Download cuda_11.8.0_520.61.05_linux.run
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```

- After downloading, you need to check the version of gcc. According to [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
```bash
gcc --version
```

- I downloaded gcc 11 by default, so I need to switch to gcc 9
```bash
sudo apt install gcc-9 g++-9
```
```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
```

- Install Cuda 11.08 after switching
```bash
sudo sh cuda_11.8.0_520.61.05_linux.run
```

- If the following output appears, select Continue
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Existing package manager installation of the driver found. It is strongly    │
│ recommended that you remove this before continuing.                          │
│ Abort                                                                        │
│ Continue                                                                     │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│ Up/Down: Move | 'Enter': Select                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

- then accept
```
┌──────────────────────────────────────────────────────────────────────────────┐
│  End User License Agreement                                                  │
│  --------------------------                                                  │
│                                                                              │
│  NVIDIA Software License Agreement and CUDA Supplement to                    │
│  Software License Agreement. Last updated: October 8, 2021                   │
│                                                                              │
│  The CUDA Toolkit End User License Agreement applies to the                  │
│  NVIDIA CUDA Toolkit, the NVIDIA CUDA Samples, the NVIDIA                    │
│  Display Driver, NVIDIA Nsight tools (Visual Studio Edition),                │
│  and the associated documentation on CUDA APIs, programming                  │
│  model and development tools. If you do not agree with the                   │
│  terms and conditions of the license agreement, then do not                  │
│  download or use the software.                                               │
│                                                                              │
│  Last updated: October 8, 2021.                                              │
│                                                                              │
│                                                                              │
│  Preface                                                                     │
│  -------                                                                     │
│                                                                              │
│──────────────────────────────────────────────────────────────────────────────│
│ Do you accept the above EULA? (accept/decline/quit):                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

- Then you need to note that the Driver has been installed previously and is checked by default. You need to uncheck it to avoid installation errors.
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 520.61.05                                                           │
│ + [X] CUDA Toolkit 11.8                                                      │
│   [X] CUDA Demo Suite 11.8                                                   │
│   [X] CUDA Documentation 11.8                                                │
│ - [ ] Kernel Objects                                                         │
│      [ ] nvidia-fs                                                           │
│   Options                                                                    │
│   Install                                                                    │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└──────────────────────────────────────────────────────────────────────────────┘
```

- After final installation, the following content is output
```
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-11.8/

Please make sure that
 -   PATH includes /usr/local/cuda-11.8/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.8/lib64, or, add /usr/local/cuda-11.8/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.8/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 520.00 is required for CUDA 11.8 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log
```

- Add the installation path `/usr/local/cuda-11.8/bin`, `/usr/local/cuda-11.8/lib64` and `/usr/local/cuda-11.8/extras/CUPTI/lib64` to the environment variables
```bash
echo "# <<< Cuda-11.08 >>> #" >> ~/.bashrc
echo "export PATH=$PATH:/usr/local/cuda-11.8/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/extras/CUPTI/lib64" >> ~/.bashrc
```

- Reload environment variables
```bash
source ~/.bashrc
```

- Verify installation
```bash
nvcc -V
```

- Output the following content to show that the configuration is successful
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```
### 2.3 Install Beagle-lib
- According to Mrbayes' INSTALL file, MrBayes uses the Beagle library if it is available.  It will make use of release 3.1.2 of Beagle but should also work with release 2.1.3 of the library. So i downloaded Beagle-3.1.2.
```bash
git clone --branch=v3.1.2 --depth=1 \
	'https://github.com/beagle-dev/beagle-lib.git'
```

- Enter Beagle-lib
```bash
cd beagle-lib
```

- Run the ./autogen.sh file
```bash
./autogen.sh
```

- After running the ./autogen.sh file, the configure file will be generated, but the corresponding NVIDIA architectures need to be modified.
- First check the NVIDIA architectures in the default configure file. The output result defaults to `NVCCFLAGS="-O3 -arch compute_30"`
```bash
grep 'NVCCFLAGS="-O3 -arch compute_' configure
```

- Next, you need to modify the corresponding architecture. The architecture of the RTX4060 graphics card is Ada, which corresponds to compute_89.
```
NVCCFLAGS="-O3 -arch compute_30"
```

- modify to
```
NVCCFLAGS="-O3 -arch compute_89"
```

- You can use the following commands to modify it, or you can use vim or other text editors to modify it.
```bash
sed -i 's/NVCCFLAGS="-O3 -arch compute_30"/NVCCFLAGS="-O3 -arch compute_89"/g' configure
```

- Before executing this command, please make sure you have sufficient permissions to modify the configure file. If not, you may need to use sudo to obtain the necessary permissions:
```bash
sudo sed -i 's/NVCCFLAGS="-O3 -arch compute_30"/NVCCFLAGS="-O3 -arch compute_89"/g' configure
```

- Next install
```bash
./configure --prefix=/usr/local/beagle
make
sudo make install
```

- Configure environment variables
```bash
echo "# <<< Beagle-lib-3.1.2 >>> #" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/beagle/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=/usr/local/beagle/lib/pkgconfig:$PKG_CONFIG_PATH" >> ~/.bashrc
source ~/.bashrc
```

### 2.4 Install Openmpi
- The installation of openmpi is relatively simple. You can install it in the following order.
```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
tar -xzvf openmpi-4.1.5.tar.gz
cd openmpi-4.1.5/
./configure --prefix=/usr/local/openmpi
make
sudo make install
echo "# <<< Openmpi-4.1.5 >>> #" >> ~/.bashrc
echo "export PATH=/usr/local/openmpi/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export MANPATH=/usr/local/openmpi/share/man:$MANPATH" >> ~/.bashrc
source ~/.bashrc
```

- Verify installation, Runs on 4 CPU cores: `mpirun -np 4`
```bash
cd openmpi-4.1.5/examples/
make
mpirun -np 4 hello_c
```

- If the following output appears, the verification is successful.
```
Hello, world, I am 2 of 4, (Open MPI v4.1.5, package: Open MPI orange@bee Distribution, ident: 4.1.5, repo rev: v4.1.5, Feb 23, 2023, 105)
Hello, world, I am 3 of 4, (Open MPI v4.1.5, package: Open MPI orange@bee Distribution, ident: 4.1.5, repo rev: v4.1.5, Feb 23, 2023, 105)
Hello, world, I am 0 of 4, (Open MPI v4.1.5, package: Open MPI orange@bee Distribution, ident: 4.1.5, repo rev: v4.1.5, Feb 23, 2023, 105)
Hello, world, I am 1 of 4, (Open MPI v4.1.5, package: Open MPI orange@bee Distribution, ident: 4.1.5, repo rev: v4.1.5, Feb 23, 2023, 105)
```

### 2.5 Install Mrbayes
- The installation of Mrbayes is as follows
```bash
wget https://github.com/NBISweden/MrBayes/releases/download/v3.2.7/mrbayes-3.2.7.tar.gz
tar -xzvf mrbayes-3.2.7.tar.gz
cd mrbayes-3.2.7/
./configure --with-mpi --with-beagle
make
sudo make install
```
## 3 Verify installation
- use 4 cores to run mrbayes
```bash
mpirun -np 4 mb
```

- Check out the available beagle libraries
```
showbeagle
```

```
                            MrBayes 3.2.7 x86_64

                      (Bayesian Analysis of Phylogeny)

                             (Parallel version)
                         (4 processors available)

              Distributed under the GNU General Public License


               Type "help" or "help <command>" for information
                     on the commands that are available.

                   Type "about" for authorship and general
                       information about the program.


MrBayes > showbeagle

   Available resources reported by beagle library:
	Resource 0:
	Name: CPU
	Flags: PROCESSOR_CPU PRECISION_DOUBLE PRECISION_SINGLE COMPUTATION_SYNCH
             EIGEN_REAL EIGEN_COMPLEX SCALING_MANUAL SCALING_AUTO
             SCALING_ALWAYS SCALING_DYNAMIC SCALERS_RAW SCALERS_LOG
             VECTOR_NONE VECTOR_SSE THREADING_NONE THREADING_CPP

	Resource 1:
	Name: NVIDIA GeForce RTX 4060 Laptop GPU
	Desc: Global memory (MB): 7915 | Clock speed (Ghz): 1.89 | Number of cores: 3072
	Flags: PROCESSOR_GPU PRECISION_DOUBLE PRECISION_SINGLE COMPUTATION_ASYNCH
             COMPUTATION_SYNCH EIGEN_REAL EIGEN_COMPLEX SCALING_MANUAL
             SCALING_AUTO SCALING_ALWAYS SCALING_DYNAMIC SCALERS_RAW
             SCALERS_LOG VECTOR_NONE THREADING_NONE

   BEAGLE version: 3.1.2
```
