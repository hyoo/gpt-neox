#!/bin/bash

#SBATCH -A MED106_crusher
#SBATCH -N 12
#SBATCH -t 04:00:00
#SBATCH -J RB_ft_26k
#SBATCH -o %x-%j.out
#SBATCH -p batch
#SBATCH --mail-user hsyoo@anl.gov
#SBATCH --mail-type END,FAIL

set +x
module load rocm/4.5.2 gcc/11.2.0

source /ccs/home/hsyoo/crusher_conda.sh
conda activate /gpfs/alpine/med106/proj-shared/hsyoo/Crusher/GPTNeoX20B/conda_env

export TORCH_EXTENSIONS_DIR=/ccs/home/hsyoo/crusher_neox/pytorch_extensions/
export MAX_JOBS=64
export HCC_AMDGPU_TARGET=gfx90a

export LD_PRELOAD="/opt/cray/pe/gcc/11.2.0/snos/lib64/libstdc++.so.6 /gpfs/alpine/world-shared/bip214/rocm_smi_lib/build/rocm_smi/librocm_smi64.so"
export LD_PRELOAD="${LD_PRELOAD} ${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so"

# hostfile
srun hostname > hostfile.txt
sed -i 's/$/ slots=8/' hostfile.txt

# train
python deepy.py train.py -d configs RB_ft_26k.yml
