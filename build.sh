#!/bin/bash

# build the fortran modules in-place
echo "Building oncvpsp_routines..."
pushd . >/dev/null
cd pstudio/oncvpsp_routines
. build.sh
popd >/dev/null



