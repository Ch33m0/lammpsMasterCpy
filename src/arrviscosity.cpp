//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//NOTES: created class that inherits Viscosity, uses ARRHENIUS style 

#include "arrviscosity.h"

using namespace LAMMPS_NS;

ViscosityArr::ViscosityArr(double C1, double C2) {
    this->C1 = C1;
    this->C2 = C2;
    this->types= 1;
    //this->style="ARR";
}


double ViscosityArr::computeViscosity(int type, double temp) {
    if (type==this->types)
        return  C1*exp(C2/temp);
}
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 