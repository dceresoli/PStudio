#!/bin/bash

rm -f *.so *.dll *.dylib
f2py -m oncvpsp -c aeo.f90 lschfb.f90 hartree.f90   # ldiracfb.f90

