//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//NOTES: created new viscosity class header file
//       just a constructor and virtual compute method... will be inherited by specific kind of visc calculations 

#ifndef LAMMPS_VISCOSITY_H
#define LAMMPS_VISCOSITY_H

#include "math.h"

namespace LAMMPS_NS {

//base class structure for any type of Viscosity that is desired to be added to source code

    class Viscosity {
    public:
    
        Viscosity();
        //computes the dynamic viscosity for a given input temperature and particle type
        virtual double computeViscosity(int type, double temperature)=0;
        //for adding new set of parameters one by one, when multiple fluid viscosites are selected (set up currently specifically for 4P exponential)
        virtual void addParams(int type, double p1, double p2, double p3, double p4) {};
        //returns the number of fluid types it is storing viscosity data for
        int getTypes();
        int types;
        
    };
}

#endif //LAMMPS_VISCOSITY_H 
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!