//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// new class for isotherm EOS

#ifdef PAIR_CLASS

PairStyle(sph/isothermal,PairSPHIsothermal)

#else

#ifndef LMP_PAIR_ISOTHERMAL_H
#define LMP_PAIR_ISOTHERMAL_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSPHIsothermal : public Pair {
 public:
  PairSPHIsothermal(class LAMMPS *);
  virtual ~PairSPHIsothermal();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual double init_one(int, int);

 protected:
  double *rho0, *soundspeed, *B, *p0;
  double **cut, **viscosity;
  int first;

  void allocate();
};

}

#endif
#endif

