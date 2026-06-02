#PBS -N test
#PBS -m abe
#PBS -M zeinab.pouramini@sirjantech.ac.ir
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q cuda7
cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=/share/apps/cuda/cuda-10.1/lib64:$LD_LIBRARY_PATH
export PATH=/share/apps/cuda/cuda-10.1/bin:$PATH
source /share/apps/Anaconda/anaconda3.8/bin/activate zeinab
python -u run.py > output.txt


