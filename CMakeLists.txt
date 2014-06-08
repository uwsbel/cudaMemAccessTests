CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(thrust_tests)
INCLUDE(SBELUtils.cmake)

enable_cuda_support()

ADD_SUBDIRECTORY(src)
