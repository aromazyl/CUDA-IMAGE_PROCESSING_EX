#
# Makefile
# zhangyule, 2017-09-28 11:53
#

all: ./possion_blending.cu ./possion_test.cu
	nvcc -o possion_test possion_test.cu


# vim:ft=make
#
