//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// NOTES:

#ifndef LAMMPS_4PVISCOSITYM_H
#define LAMMPS_4PVISCOSITYM_H

#include "viscosity.h"

namespace LAMMPS_NS{

class Viscosity4PM  : public Viscosity {

public:

    Viscosity4PM(int types);
    void addParams(int type, double p1, double p2, double p3, double p4);
    double computeViscosity(int type, double temperature);


private:

    double ** params;
};

}

#endif //LAMMPS_4PVISCOSITY_H