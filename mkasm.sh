#!/bin/sh
#
# In the build directory, for unix, peek at generated assembler code
#
make `make help | grep '[.]s$' | cut -d' ' -f2`

