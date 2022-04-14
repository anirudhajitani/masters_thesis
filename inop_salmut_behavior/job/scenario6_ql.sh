#!/bin/bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --account=def-adityam

source ~/offload-env/bin/activate
module load python/3.6
cd ~/scratch/tccn/masters_thesis/inop_salmut_behavior/
export PYTHONPATH='.'
export LC_ALL=en_CA.utf8
export LANG=en_CA.utf8
#cd examples/tree-topology/
python main_train.py --algo 2 --folder scenario_6_lambd_2 --env_name q_learning --logdir ql_log --lambd 0.5 --lambd_evolve True --user_identical False --user_evolve True
