 //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// NOTES:

#ifndef LAMMPS_ARRVISCOSITY_H
#define LAMMPS_ARRVISCOSITY_H

#include "viscosity.h"

namespace LAMMPS_NS{

//class that stores parameters and computes Arrhenius-stype temperature-dependent dynamic viscosity for single fluid type

class ViscosityArr  : public Viscosity {

public:

    ViscosityArr(double C1, double C2);
    double computeViscosity(int type, double temperature);


private:

    double C1, C2;
};

}

#endif //LAMMPS_ARRVISCOSITY_H
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111