//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//NOTES: created class that inherits Viscosity....
// ALL FLUID PARTICLES MUST BE FROM 1 to n-1 TYPE.... nth TYPE IS THE BOUNDARY PARTICLES AND IS NOT COUNTED

#include "4PviscosityM.h"

using namespace LAMMPS_NS;

Viscosity4PM::Viscosity4PM(int types) {
    this->types=types;
    this->params = new double*[types];
    for(int i = 0; i < types; ++i){
      params[i] = new double[4];
    }  
}


void Viscosity4PM::addParams(int type, double p1, double p2, double p3, double p4){
    this->params[type-1][0]= p1;
    this->params[type-1][1]= p2;
    this->params[type-1][2]= p3;
    this->params[type-1][3]= p4;

}

double Viscosity4PM::computeViscosity(int type, double temp) {
  if(type<=this->types)
    return  this->params[type-1][0]*exp(this->params[type-1][1]/temp + this->params[type-1][2]*temp + this->params[type-1][3]*temp*temp);
  else
    return 0.0;
} 
