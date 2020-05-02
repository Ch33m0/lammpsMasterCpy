//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//NOTES: created new viscosity class header file
//       just a constructor and virtual compute method... will be inherited by specific kind of visc calculations 

#ifndef LAMMPS_VISCOSITY_H
#define LAMMPS_VISCOSITY_H

#include "math.h"

namespace LAMMPS_NS {

    class Viscosity {
    public:
    
        Viscosity();
        virtual double computeViscosity(int type, double temperature)=0;
        virtual void addParams(int type, double p1, double p2, double p3, double p4) {};
        int getTypes();
        int types;
        
    };
}




#endif //LAMMPS_VISCOSITY_H 
