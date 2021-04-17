#!/bin/bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --account=def-adityam

source ~/offload-env/bin/activate
module load python/3.6
cd ~/scratch/inop_new
export PYTHONPATH='.'
export LC_ALL=en_CA.utf8
export LANG=en_CA.utf8
#cd examples/tree-topology/
python main_eval2.py --algo 3 --folder scenario_4_2_lambd_2 --env_name salmut --logdir s1 --lambd 0.5 --lambd_evolve False --user_identical False --user_evolve False --start_iter 333 --step 333
