 //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// NOTES:

#ifndef LAMMPS_ARRVISCOSITY_H
#define LAMMPS_ARRVISCOSITY_H

#include "viscosity.h"

namespace LAMMPS_NS{

class ViscosityArr  : public Viscosity {

public:

    ViscosityArr(double C1, double C2);
    double computeViscosity(int type, double temperature);


private:

    double C1, C2;
};

}

#endif //LAMMPS_ARRVISCOSITY_H
