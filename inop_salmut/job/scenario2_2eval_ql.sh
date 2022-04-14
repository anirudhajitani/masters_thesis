#!/bin/bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --account=def-adityam

source ~/offload-env/bin/activate
module load python/3.6
cd ~/scratch/tccn/masters_thesis/inop_salmut/
export PYTHONPATH='.'
export LC_ALL=en_CA.utf8
export LANG=en_CA.utf8
#cd examples/tree-topology/
python main_eval2.py --algo 2 --folder scenario_2_lambd_2 --env_name q_learning --logdir ql_log --lambd 0.5 --lambd_evolve True --user_identical True --user_evolve False --start_iter 333 --step 333
