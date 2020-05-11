//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#ifdef NV_KERNEL
#include "lal_aux_fun1.h"
#ifndef _DOUBLE_DOUBLE
texture<float4> pos_tex;
#else
texture<int4,1> pos_tex;
#endif
#else
#define pos_tex x_
#endif

#define MY_PI (acctyp)3.14159265358979323846

__kernel void k_sph_taitwater(const __global numtyp4 *restrict x_,
                     const __global numtyp4 *restrict coeff,
                     const int lj_types,
                     const __global numtyp *restrict sp_lj_in,
                     const __global int *dev_nbor,
                     const __global int *dev_packed,
                     __global acctyp4 *restrict ans,
                     __global acctyp *restrict engv,
                     const int eflag, const int vflag, const int inum,
                     const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  
  //CP: needs to be modified to have more atom info for SPH
  
  atom_info(t_per_atom,ii,tid,offset);

//CP: sp_lj and sp_lj_in would not be used for SPH taitwater
  __local numtyp sp_lj[4];
  sp_lj[0]=sp_lj_in[0];
  sp_lj[1]=sp_lj_in[1];
  sp_lj[2]=sp_lj_in[2];
  sp_lj[3]=sp_lj_in[3];

  acctyp energy=(acctyp)0;
  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp virial[6];
  for (int i=0; i<6; i++)
    virial[i]=(acctyp)0;

  if (ii<inum) {
    int nbor, nbor_end;
    int i, numj;
    __local int n_stride;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

//CP: would need a fetch4 for i atom velocity vector as well
    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    int itype=ix.w;


//CP: here would need to access rho for computer Tait EOS force of i atom 

    numtyp factor_lj;
    for ( ; nbor<nbor_end; nbor+=n_stride) {

      int j=dev_packed[nbor];
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;

//CP: would need a fetch4 for j atom velocity vector as well
      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      int jtype=jx.w;

      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp rsq = delx*delx+dely*dely+delz*delz;
      
//CP: below here, this would need to be all different, in style of SPH Taitwater,
// would need to have fj EOS computation, then the artificial viscosity calculations, then different equation for change in force
//CP: also need to add equation to change density and thermal energy of system, then add/modify all methods that store this data back from GPU

      int mtype=itype*lj_types+jtype;
      if (rsq<coeff[mtype].z) {
        numtyp force;
        numtyp r = ucl_sqrt(rsq);
        numtyp arg = MY_PI*r/coeff[mtype].y;
        if (r > (numtyp)0.0) force = factor_lj * coeff[mtype].x *
                       sin(arg) * MY_PI/coeff[mtype].y*ucl_recip(r);
        else force = (numtyp)0.0;

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (eflag>0) {
          numtyp e=coeff[mtype].x * ((numtyp)1.0+cos(arg));
          energy+=factor_lj*e;
        }
        if (vflag>0) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor
    store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                  ans,engv);
  } // if ii
}

//CP: removed fast version, no point in keeping it
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!