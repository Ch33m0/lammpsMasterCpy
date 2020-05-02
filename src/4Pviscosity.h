//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// NOTES:

#ifndef LAMMPS_4PVISCOSITY_H
#define LAMMPS_4PVISCOSITY_H

#include "viscosity.h"

namespace LAMMPS_NS{

class Viscosity4P  : public Viscosity {

public:

    Viscosity4P(double p1, double p2, double p3, double p4);
    double computeViscosity(int type, double temperature);


private:

    double p1, p2, p3, p4;
};

}

#endif //LAMMPS_4PVISCOSITY_H