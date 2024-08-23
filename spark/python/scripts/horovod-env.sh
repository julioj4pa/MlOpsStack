#!/usr/bin/env bash

#NCCL_INCLUDE_DIR="/usr/include/" NCCL_LIB_DIR="/usr/lib/" USE_SYSTEM_NCCL=1
#export PATH=$PATH:/home/spark/ucx-1.17.0/install/bin:/home/spark/openmpi-5.0.3/install/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/spark/openmpi-5.0.3/ompi/include
#export NCCL_INCLUDE_DIR="/usr/include/"
#export NCCL_LIB_DIR="/usr/lib/x86_64-linux-gnu/"
#export USE_SYSTEM_NCCL=1
#HOROVOD_DEBUG - {1}. Install a debug build of Horovod with checked assertions, disabled compiler optimizations etc.
export HOROVOD_DEBUG=1
#HOROVOD_BUILD_ARCH_FLAGS - additional C++ compilation flags to pass in for your build architecture.

#HOROVOD_CUDA_HOME - path where CUDA include and lib directories can be found.
export HOROVOD_CUDA_HOME=/usr/local/cuda-12.5
#HOROVOD_BUILD_CUDA_CC_LIST - List of compute capabilities to build Horovod CUDA kernels for (example: HOROVOD_BUILD_CUDA_CC_LIST=60,70,75) https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQuery
export HOROVOD_BUILD_CUDA_CC_LIST=86
#HOROVOD_NCCL_HOME - path where NCCL include and lib directories can be found.

#HOROVOD_NCCL_INCLUDE - path to NCCL include directory.
export HOROVOD_NCCL_INCLUDE=/usr/include/
#HOROVOD_NCCL_LIB - path to NCCL lib directory.
export HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu/
#HOROVOD_NCCL_LINK - {SHARED, STATIC}. Mode to link NCCL library. Defaults to STATIC for CUDA, SHARED for ROCm.
export HOROVOD_NCCL_LINK=SHARED
#HOROVOD_WITH_GLOO - {1}. Require that Horovod is built with Gloo support enabled.
export HOROVOD_WITH_GLOO=0
#HOROVOD_WITHOUT_GLOO - {1}. Skip building with Gloo support.
export HOROVOD_WITHOUT_GLOO=1
#HOROVOD_WITH_MPI - {1}. Require that Horovod is built with MPI support enabled.
export HOROVOD_WITH_MPI=1
#HOROVOD_WITHOUT_MPI - {1}. Skip building with MPI support.
export HOROVOD_WITHOUT_MPI=0
#HOROVOD_GPU - {CUDA, ROCM}. Framework to use for GPU operations.
export HOROVOD_GPU=CUDA
#HOROVOD_GPU_OPERATIONS - {NCCL, MPI}. Framework to use for GPU tensor allreduce, allgather, and broadcast.
export HOROVOD_GPU_OPERATIONS=NCCL
#HOROVOD_GPU_ALLREDUCE - {NCCL, MPI}. Framework to use for GPU tensor allreduce.
#export HOROVOD_GPU_ALLREDUCE=NCCL
#HOROVOD_GPU_ALLGATHER - {NCCL, MPI}. Framework to use for GPU tensor allgather.
#export HOROVOD_GPU_ALLGATHER=NCCL
#HOROVOD_GPU_BROADCAST - {NCCL, MPI}. Framework to use for GPU tensor broadcast.
#export HOROVOD_GPU_BROADCAST=NCCL
#HOROVOD_GPU_ALLTOALL - {NCCL, MPI}. Framework to use for GPU tensor alltoall.
#export HOROVOD_GPU_ALLTOALL=NCCL
#HOROVOD_GPU_REDUCESCATTER - {NCCL, MPI}. Framework to use for GPU tensor reducescatter.
#export HOROVOD_GPU_REDUCESCATTER=NCCL
#HOROVOD_ALLOW_MIXED_GPU_IMPL - {1}. Allow Horovod to install with NCCL allreduce and MPI GPU allgather / broadcast / alltoall / reducescatter. Not recommended due to a possible deadlock.
#export HOROVOD_ALLOW_MIXED_GPU_IMPL=0

#export MPI_INCLUDE_PATH="/home/spark/openmpi-5.0.3/ompi/include"
#export OPAL_PREFIX="/home/spark/openmpi-5.0.3/install"
#HOROVOD_CPU_OPERATIONS - {MPI, GLOO, CCL}. Framework to use for CPU tensor allreduce, allgather, and broadcast.
export HOROVOD_CPU_OPERATIONS=MPI
#HOROVOD_CMAKE - path to the CMake binary used to build Horovod. Installed version cmake version 3.22.1
export HOROVOD_CMAKE=/usr/bin/cmake
#HOROVOD_WITH_TENSORFLOW - {1}. Require Horovod to install with TensorFlow support enabled.
export HOROVOD_WITH_TENSORFLOW=1
#HOROVOD_WITHOUT_TENSORFLOW - {1}. Skip installing TensorFlow support.
export HOROVOD_WITHOUT_TENSORFLOW=0
#HOROVOD_WITH_PYTORCH - {1}. Require Horovod to install with PyTorch support enabled.
export HOROVOD_WITH_PYTORCH=1
#HOROVOD_WITHOUT_PYTORCH - {1}. Skip installing PyTorch support.
export HOROVOD_WITHOUT_PYTORCH=0
#HOROVOD_WITH_MXNET - {1}. Require Horovod to install with MXNet support enabled.
export HOROVOD_WITH_MXNET=1
#HOROVOD_WITHOUT_MXNET - {1}. Skip installing MXNet support.
export HOROVOD_WITHOUT_MXNET=0

#HOROVOD_NCCL_LINK=SHARED HOROVOD_NCCL_INCLUDE=/usr/include HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_OPERATIONS=NCCL python setup.py bdist_wheel