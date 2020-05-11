//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//changed
// made by Callum Pollock
// gpu version of standard pair_sph_taitwater 

#include "pair_sph_taitwater_gpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "integrate.h"
#include "memory.h"
#include "error.h"
#include "neigh_request.h"
#include "universe.h"
#include "update.h"
#include "domain.h"
#include "gpu_extra.h"
#include "math_const.h"
#include "suffix.h"

using namespace LAMMPS_NS;

// External functions from cuda library for atom decomposition

//CP: parameters would need to be changed to match changes in "lal_sph_taitwater_ext" and "lal_sph_taitwater", and code in this file would then need to call 
// reference different values to match these paramter needs

int sph_taitwater_gpu_init(const int ntypes, double **cutsq, double **prefactor,
                   double **cut, double *special_lj, const int nlocal,
                   const int nall, const int max_nbors, const int maxspecial,
                   const double cell_size, int &gpu_mode, FILE *screen);
void sph_taitwater_gpu_reinit(const int ntypes, double **cutsq, double **host_prefactor,
                     double **host_cut);
void sph_taitwater_gpu_clear();
int ** sph_taitwater_gpu_compute_n(const int ago, const int inum,
                           const int nall, double **host_x, int *host_type,
                           double *sublo, double *subhi, tagint *tag, int **nspecial,
                           tagint **special, const bool eflag, const bool vflag,
                           const bool eatom, const bool vatom, int &host_start,
                           int **ilist, int **jnum,
                           const double cpu_time, bool &success);
void sph_taitwater_gpu_compute(const int ago, const int inum, const int nall,
                       double **host_x, int *host_type, int *ilist, int *numj,
                       int **firstneigh, const bool eflag, const bool vflag,
                       const bool eatom, const bool vatom, int &host_start,
                       const double cpu_time, bool &success);
double sph_taitwater_gpu_bytes();


using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairSPHTaitwaterGPU::PairSPHTaitwaterGPU(LAMMPS *lmp) : PairSPHTaitwater(lmp), gpu_mode(GPU_FORCE)
{
  respa_enable = 0;
  cpu_time = 0.0;
  suffix_flag |= Suffix::GPU;
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/* ----------------------------------------------------------------------
   free all arrays
------------------------------------------------------------------------- */

PairSPHTaitwaterGPU::~PairSPHTaitwaterGPU()
{
  sph_taitwater_gpu_clear();
}

/* ---------------------------------------------------------------------- */

void PairSPHTaitwaterGPU::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  int nall = atom->nlocal + atom->nghost;
  int inum, host_start;

  bool success = true;
  int *ilist, *numneigh, **firstneigh;
  if (gpu_mode != GPU_FORCE) {
    inum = atom->nlocal;
    firstneigh = sph_taitwater_gpu_compute_n(neighbor->ago, inum, nall,
                                     atom->x, atom->type, domain->sublo,
                                     domain->subhi, atom->tag, atom->nspecial,
                                     atom->special, eflag, vflag, eflag_atom,
                                     vflag_atom, host_start,
                                     &ilist, &numneigh, cpu_time, success);
  } else {
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    sph_taitwater_gpu_compute(neighbor->ago, inum, nall, atom->x, atom->type,
                      ilist, numneigh, firstneigh, eflag, vflag, eflag_atom,
                      vflag_atom, host_start, cpu_time, success);
  }
  if (!success)
    error->one(FLERR,"Insufficient memory on accelerator");

  if (host_start<inum) {
    cpu_time = MPI_Wtime();
    cpu_compute(host_start, inum, eflag, vflag, ilist, numneigh, firstneigh);
    cpu_time = MPI_Wtime() - cpu_time;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSPHTaitwaterGPU::init_style()
{
  if (force->newton_pair)
    error->all(FLERR,"Cannot use newton pair with sph/taitwater/gpu pair style");

  // Repeat cutsq calculation because done after call to init_style
  double maxcut = -1.0;
  double mcut;
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
        mcut = init_one(i,j);
        mcut *= mcut;
        if (mcut > maxcut)
          maxcut = mcut;
        cutsq[i][j] = cutsq[j][i] = mcut;
      } else
        cutsq[i][j] = cutsq[j][i] = 0.0;
    }
  }
  double cell_size = sqrt(maxcut) + neighbor->skin;

  int maxspecial=0;
  if (atom->molecular)
    maxspecial=atom->maxspecial;
    
  int success = sph_taitwater_gpu_init(atom->ntypes+1, cutsq, prefactor, cut,
                              force->special_lj, atom->nlocal,
                              atom->nlocal+atom->nghost, 300, maxspecial,
                              cell_size, gpu_mode, screen);
  GPU_EXTRA::check_flag(success,error,world);

  if (gpu_mode == GPU_FORCE) {
    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
  }
}

/* ---------------------------------------------------------------------- */

void PairSPHTaitwaterGPU::reinit()
{
  Pair::reinit();

  sph_taitwater_gpu_reinit(atom->ntypes+1, cutsq, prefactor, cut);
}

/* ---------------------------------------------------------------------- */

double PairSPHTaitwaterGPU::memory_usage()
{
  double bytes = Pair::memory_usage();
  return bytes + sph_taitwater_gpu_bytes();
}

/* ---------------------------------------------------------------------- */

void PairSPHTaitwaterGPU::cpu_compute(int start, int inum, int eflag, int /* vflag */,
                              int *ilist, int *numneigh, int **firstneigh) {
  int i,j,ii,jj,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double r,rsq,arg,factor_lj;
  int *jlist;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  double *special_lj = force->special_lj;

  // loop over neighbors of my atoms

  for (ii = start; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        arg = MY_PI*r/cut[itype][jtype];
        if (r > 0.0) fpair = factor_lj * prefactor[itype][jtype] *
                       sin(arg) * MY_PI/cut[itype][jtype]/r;
        else fpair = 0.0;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;

        if (eflag)
          evdwl = factor_lj * prefactor[itype][jtype] * (1.0+cos(arg));

        if (evflag) ev_tally_full(i,evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }
}
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!