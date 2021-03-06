//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//changed
#ifdef PAIR_CLASS

PairStyle(sph/taitwater/gpu,PairSPHTaitwaterGPU)

#else

#ifndef LMP_PAIR_SPH_TAITWATER_GPU_H
#define LMP_PAIR_SPH_TAITWATER_GPU_H

#include "pair_sph_taitwater.h"

namespace LAMMPS_NS {

class PairSPHTaitwaterGPU : public PairSPHTaitwater {
 public:
  PairSPHTaitwaterGPU(LAMMPS *lmp);
  ~PairSPHTaitwaterGPU();
  void cpu_compute(int, int, int, int, int *, int *, int **);
  void compute(int, int);
  void init_style();
  void reinit();
  double memory_usage();

 enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

 private:
  int gpu_mode;
  double cpu_time;
};

}
#endif
#endif

/* ERROR/WARNING messages:
E: Insufficient memory on accelerator
There is insufficient memory on one of the devices specified for the gpu
package
E: Cannot use newton pair with soft/gpu pair style
Self-explanatory.

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!