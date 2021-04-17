#!/bin/bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --account=def-adityam

source ~/offload-env/bin/activate
module load python/3.6
cd ~/scratch/inop_new
export PYTHONPATH='.'
export LC_ALL=en_CA.utf8
export LANG=en_CA.utf8
#cd examples/tree-topology/
python main_train.py --algo 3 --folder scenario_1_2_lambd_2 --env_name salmut --logdir salmut_log --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False
