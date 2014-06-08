cuda_add_executable(driver basic.cu defaultTest.cu driver.cu hostSide.cpp
	pinned.cu unifiedMem.cu uvaTest.cu zerocopy.cu)
