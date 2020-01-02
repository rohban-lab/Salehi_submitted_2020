#PBS -N myjob
#PBS -m abe
#PBS -M your@email.adress
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q tensorflow

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH

cd $PBS_O_WORKDIR
/state/partition1/anaconda3/bin/python3.6 -u v1.py --gpu_id "0" > output.txt
