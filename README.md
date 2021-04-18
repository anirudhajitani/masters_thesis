# masters_thesis
Intelligent Node Overload Protection using Reinforcement Learning

Requirements 

1.	Need a system with at least python3.6
2.	In order to run docker experiments we need to have docker installed with user permissions.
a.	https://docs.docker.com/engine/install/ubuntu/ 
b.	https://docs.docker.com/engine/reference/commandline/docker/ (Useful Docker commands)
3.	Python packages and dependencies present in requirements.txt
a.	pip install -r requirements.txt


Setup & Configuration (Docker testbed)

1.	Go to inop_cloud folder
2.	Make sure to execute the commands below before running other docker images.
3.	Execute the following commands:
a.	docker build --tag inop .
b.	docker run -d --name=app1 -p 4000:3333 inop
4.	Go to docker_dummy subfolder
5.	Execute the following commands:
a.	docker build --tag inop_dummy .
b.	docker run -d --name=app2 -p 5000:3333 inop_dummy
6.	Come out of the docker_dummy folder to main folder (inop_cloud)
7.	Execute the following commands to check if the IP address of the edge nodes match the default IP address:
a.	docker inspect -f '{{ .NetworkSettings.IPAddress }}' app1 (should be 172.17.0.2)
b.	docker inspect -f '{{ .NetworkSettings.IPAddress }}' app2 (should be 172.17.0.3)

Running a Scenario

A. Computer Simulations

Training

(SALMUT)

Scenario 1
python main_train.py --algo 3 --folder scenario_1_lambd_2 --env_name salmut --logdir salmut_log --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False

Scenario 2
python main_train.py --algo 3 --folder scenario_2_lambd_2 --env_name salmut --logdir salmut_log_2 --lambd 0.5 --lambd_evolve True --user_identical True --user_evolve False

Scenario 3
python main_train.py --algo 3 --folder scenario_3_lambd_2 --env_name salmut --logdir salmut_log_3 --lambd 0.5 --lambd_evolve True --user_identical False --user_evolve False

Scenario 4
python main_train.py --algo 3 --folder scenario_4_lambd_2 --env_name salmut --logdir salmut_log_4 --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve True

Scenario 5
python main_train.py --algo 3 --folder scenario_5_lambd_2 --env_name salmut --logdir salmut_log_5 --lambd 0.5 --lambd_evolve True --user_identical True --user_evolve True

Scenario 6
python main_train.py --algo 3 --folder scenario_6_lambd_2 --env_name salmut --logdir salmut_log_6 --lambd 0.5 --lambd_evolve True --user_identical False --user_evolve True

For PPO and A2C, replace –algo with 0 and 1 respectively, and –env_name with ppo and a2c and –logdir names accordingly.

Evaluation (This can be run in parallel or serially. Can only be performed once training is done.)

Let’s look at an example of parallel evaluation of eval for SALMUT.

python main_eval2.py --algo 3 --folder scenario_1_lambd_2 --env_name salmut --logdir s1 --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False --start_iter 0 --step 333

python main_eval2.py --algo 3 --folder scenario_1_lambd_2 --env_name salmut --logdir s1 --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False --start_iter 333 --step 333

python main_eval2.py --algo 3 --folder scenario_1_lambd_2 --env_name salmut --logdir s1 --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False --start_iter 666 --step 334

Similarly, we run it for different scenarios by changing the values of (user_identical, user_evolve, and lambda_evolve).

We can also run similar experiments for PPO and A2C by changing algo, env_name, and logdir similar to training. 

Planning

We run the planning algorithms (DP and baseline) using the following commands:

Dynamic Programming Solution (algo = 0)
python main_plan.py --algo 0 --folder scenario_1_lambd_2 --env_name plan_eval --logdir plan_log_s1 --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False

Baseline Policy (algo = 1)
python main_plan.py --algo 1 --folder scenario_1_lambd_2 --env_name thres_eval --logdir thres_log_s1 --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False


B. Behavioral Analysis (Computer Simulations)

Only need to run training, commands are the exact same as the training for computer simulations. Only thing is we need to execute them in inop_salmut_behavior folder. 

C. Docker testbed

Each experiment needs to be run in separate VM. Execute the command in screen. We run only for SALMUT and Baseline. The buffers folder for each scenario should have the files N.npy and lambda.npy.

scenario_1_lambd_2/buffers/N.npy
scenario_1_lambd_2/buffers/lambda.npy

Training:

python3.6 load_gen.py scenario_1_lambd_2 salmut

python3.6 load_gen.py scenario_2_lambd_2 salmut

and so on …

Evaluation 

Change app2.py with app2_eval.py in Dockerfile and perform the steps in Configuration & Setup. Ensure all training data is present in buffers, if we had run different seeds of training in separate VMs copy all the data to all the machines, we are going to run evaluation in. 

Make sure of the following:
1.	Stop and delete any existing docker images that may be running by using:
a.	docker stop <container-name>
b.	docker rm <container-name>
2.	The results folder in the scenario_folder is empty unless the goal is to restart training from previously stopped iteration. 
3.	If we restart training, make sure to restart the edge node using:
a.	docker restart app1

python3.6 load_gen_eval.py scenario_1_lambd_2 salmut_eval

Baseline

Change app2.py with app2_baseline.py in Dockerfile and perform the steps in Configuration & Setup. 

python3.6 load_gen_.py baseline.py scenario_1_lambd_2 salmut_eval

If you are running the experiments using python command directly instead of python3.6, replace python3.6 by python or python3 in load_gen.py, load_gen_eval.py and load_gen_baseline.py in the run_rl_module_and_notify() function.

Generating Results

A. Computer Simulations

If we run evaluations in parallel, combine the results of the experiments before we plot the results:

python combine_result.py <scenario-folder> <algorithm>

Ex:
python combine_result.py scenario_1_lambd_2 salmut

Plotting results (ensure file names are same as in results folder of the scenario)

python plot_results.py scenario_1_lambd_2 

The results are in the scenario folder under results directory with name {folder}_reward.png, ex: scenario_1_lambd_2_results.png

B. Computer Simulations (Behavioral Analysis)

Plot the results
python plot_results.py scenario_1_lambd_2 

The results are in the scenario folder under results directory with name {folder}_offload_new.png and {folder}_overload_new.png.

python plot_pareto.py scenario_1_lambd_2

The results are in the scenario folder under results directory with names pareto_{folder}.png


C. Docker Testbed

If we run only training and want to see the results. All the buffers files of the different training runs needs to be in the same scenario folder.

Combine the results using

python combine_results.py scenario_1_lambd_2

Plot results using (plots offload, overload counts as well)

python plot_results_combined.py scenario_1_lambd_2

The results are in the scenario folder under results directory with names {folder}_dis_reward_testbed.png, {folder}_offload_testbed.png, and {folder}_overload_testbed.png. 

python plot_pareto.py scenario_1_lambd_2

The results are in the scenario folder under results directory with names pareto_testbed_{folder}.png


