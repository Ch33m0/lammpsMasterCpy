//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//NOTES: created class that inherits Viscosity, uses 4 parameters

#include "4Pviscosity.h"

using namespace LAMMPS_NS;

Viscosity4P::Viscosity4P(double p1, double p2, double p3, double p4) {
    this->p1 = p1;
    this->p2 = p2;
    this->p3 = p3;
    this->p4 = p4;
    this->types= 1;
}


double Viscosity4P::computeViscosity(int type, double temp) {
    if (type==this->types)
        return  p1*exp(p2/temp + p3*temp + p4*temp*temp);
} 

