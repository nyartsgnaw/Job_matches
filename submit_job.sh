#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 
#SBATCH -n 1 
#SBATCH --time=0-24:00:00 
# SBATCH --workdir=/home/straynwang/HLAGAN
#SBATCH --output=./slurm-%j.out
#SBATCH --error=./slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=straynwang@gwu.edu


# . /usr/share/Modules/init/sh
module load git/1.8.3.1
module load openmpi/1.8/gcc/4.7/cpu
module load gcc
module load slurm
module load anaconda/4.3.1 cuda/toolkit/9.0 cuda/libs/cudnn-7005
# export OMP_NUM_THREADS=1
# echo "SLURM_NODELIST $SLURM_NODELIST"

source /c1/libs/glibc/2.17/site-packages/bin/activate;

#/c1/libs/glibc/2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path /c1/libs/glibc/2.17/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:/usr/lib /c1/apps/anaconda/4.3.1/bin/python ./test.py;
#/c1/libs/glibc/2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path /c1/libs/glibc/2.17/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:/usr/lib /c1/apps/anaconda/4.3.1/bin/python ./lib/train_fasttext2.py;
/c1/libs/glibc/2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path /c1/libs/glibc/2.17/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:/usr/lib /c1/apps/anaconda/4.3.1/bin/python ./main.py;
#/c1/libs/glibc/2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path /c1/libs/glibc/2.17/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:/usr/lib /c1/apps/anaconda/4.3.1/bin/python ./analysis/compare/testGAN/init_exp_logs.py;
#/c1/libs/glibc/2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path /c1/libs/glibc/2.17/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:/usr/lib /c1/apps/anaconda/4.3.1/bin/python ./analysis/compare/testGAN/run_testGAN.py -f 'exp_logs_TG1.xlsx';
#/c1/libs/glibc/2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path /c1/libs/glibc/2.17/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:/usr/lib /c1/apps/anaconda/4.3.1/bin/python ./analysis/compare/testGAN/run_testGAN.py -f 'exp_logs_TB1.xlsx';
#/c1/libs/glibc/2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path /c1/libs/glibc/2.17/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:/usr/lib /c1/apps/anaconda/4.3.1/bin/python ./analysis/compare/testGAN/run_testGAN.py -f 'exp_logs_TD2.xlsx';
#/c1/libs/glibc/2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path /c1/libs/glibc/2.17/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:/usr/lib /c1/apps/anaconda/4.3.1/bin/python ./analysis/compare/testGAN/find_best_setup.py;
#/c1/libs/glibc/2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path /c1/libs/glibc/2.17/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:/usr/lib /c1/apps/anaconda/4.3.1/bin/python ./analysis/compare/testGAN/get_compare_df.py;



#/c1/libs/glibc/2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path /c1/libs/glibc/2.17/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:/usr/lib /c1/apps/anaconda/4.3.1/bin/python ./analysis/reduce_dim/reduce_dim.py;


