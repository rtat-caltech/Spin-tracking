#!/bin/bash

# This script is mainly here to remind you what the CMake commands are.
# But you can also use it to run the CMake commands for you, if you'd like.
# You don't have to build the cuda and cpu in build/cuda and build/cpu respectively,
# but try to keep things inside the "build" directory.
# Otherwise you might clutter the Git repo.

# The executables are also built inside the build directory, i.e. the CUDA executable is
# build/cuda/main

########################################
# Configuring CMake for the first time #
########################################

# You should only ever need to run these commands once each.
# These configure CMake, (which is to say they put a bunch of cache files in the appropriate
# build directory, so CMake knows what's happening for the future)

cmake -DCMAKE_CUDA_ARCHITECTURES=60 -S . -B build/cuda # CUDA

cmake -S . -B build/cpu #CPU

################
# Debug Builds #
################

# For debug builds, add the flag -DCMAKE_BUILD_TYPE=Debug so for example,

cmake -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Debug -S . -B build/cuda_debug # CUDA

#########################
# All subsequent builds #
#########################

# Run these whenever you want to recompile.
# As I understand it, this command does two things:
# 1. Check to see if the Makefile needs to be modified (and if so, modify it)
# 2. Run the Makefile.

cmake --build build/cuda #CUDA

cmake --build build/cpu #CPU
