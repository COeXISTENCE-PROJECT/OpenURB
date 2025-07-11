# Forked from:
<p align="center">
  <img src="docs/urb.png" align="center" width="30%"/>
</p>

---

# OpenURB

There is a microscopic traffic network in which two agent populations, human-driven vehicles (HDVs) and autonomous vehicles (AVs), make **single-step** route-choice decisions. At each departure time, a driver observes current traffic conditions, selects one of *n* routes from its origin to its destination, receives a travel-time-based reward, and the episode ends for that driver. HDVs aim to optimize individual travel time, while AVs **collectively** try to minimize group travel time.

Unlike static formulations, population composition is **dynamic**: humans may switch to AVs, or AV owners may revert to HDVs with some predefined probabilities. This creates a **non-stationary**, **variable-sized** multi-agent system where each action’s impact and optimality depend on the evolving mix of agents. Therefore, a good solution must handle the appearance of new cooperators and the elimination of known teammates, successfully scaling at **open environments**.

The task is to develop and evaluate RL methods that enable AV agents to learn robust, high-performing routing policies **under these switching dynamics,** while accounting for large group sizes and interactions with human drivers. The ultimate objective of this research is to **(i)** find the problem setting where the current approaches come short, **(ii)** quantify their shortcomings with relevant (potentially novel) dynamic cooperation performance metrics, and **(iii)** contribute with methodological extensions.

### Keywords

`MARL`, `route choice`, `open environment`, `ad-hoc teamwork`, `few-shot cooperation`

## 🔗 Workflow

`OpenURB` (similar to `URB`):
* Runs an experiment script using the `TrafficEnvironment` from `RouteRL`,
* With a RL algorithm or a baseline method (from `baseline_models/`),
* Opens algorithm, environment and task configuration files from `config/`,
* Loads the network and demand from `networks`
* Executes a typical `RouteRL` routine of
   * first learning of human drivers,
   * which then 'mutate` to CAVs,
   * are trained to optimize routing policies with the implemented algorithm,
   * THEN possibly simulating dynamic switches between humans and AVs.
* When the training is finished, it uses raw results to compute a wide-set of KPIs.

---

## 📦 Setup

#### Prerequisites 

Make sure you have SUMO installed in your system. This procedure should be carried out separately, by following the instructions provided [here](https://sumo.dlr.de/docs/Installing/index.html).

#### Cloning repository

Clone the **OpenURB** repository from github by

```bash
git clone https://github.com/COeXISTENCE-PROJECT/OpenURB.git
```

#### Creating enviroment

- **Option 1** (Recommended): Create a virtual enviroment with `venv`:

```bash
python -m venv .venv
```

and then install dependencies by:

```bash
cd URB
pip install --force-reinstall --no-cache-dir -r requirements.txt
```

## 🔬 Running experiments

#### Usage of **OpenURB** for Reinforcement Learning algorithms

To use **URB** while using RL algorithm, you have to provide in the command line the following command:

```bash
python scripts/<script_name>.py --id <exp_id> --alg-conf <hyperparam_id> --env-conf <env_conf_id> --task-conf <task_id> --net <net_name> --env-seed <env_seed> --torch-seed <torch_seed>
```

where

- ```<scipt_name>``` is the script you wish to run, available scripts are ```iql```, ```ippo```, ```open_iql```, ```cond_open_iql``` and ```parallel_open_iql```,
- ```<exp_id>``` is your own experiment identifier, for instance ```random_ing```, 
- ```<hyperparam_id>``` is the hyperparameterization identifier, it must correspond to a `.json` filename (without extension) in [`config/algo_config`](config/algo_config/). Provided scripts automatically select the algorithm-specific subfolder in this directory.
- ```<env_conf_id>``` is the environment configuration identifier. It must correspond to a `.json` filename (without extension) in [`config/env_config`](config/env_config/). It is used to parameterize environment-specific processes, such as path generation, disk operations, etc. It is **optional** and by default is set to `config1`.
- ```<task_id>``` is the task configuration identifier. It must correspond to a `.json` filename (without extension) in [`config/task_config`](config/task_config/). It is used to parameterize the simulated scenario, such as portion of AVs, duration of human learning, AV behavior, etc.
- ```<net_name>``` is the name of the network you wish to use. Must be one of the folder names in ```networks/``` i.e. ```ing_small```, ```ingolstadt_custom```, ```nangis```, ```nemours```, ```provins``` or ```saint_arnoult```,
- ```<env_seed>``` is reproducibility random seed for the traffic environment, it is **optional** and by default is set to 42,
- ```<torch_seed>``` is reproducibility random seed for PyTorch, it is **optional** and by default is set to 42.

For example, the following command runs an experiment using:
- IQL algorithm, hyperparameterized by `config/algo_config/iql/config1.json`, 
- The task specified in `config/task_config/config4.json`,
- The environment parameterization specified in `config/env_config/config1.json` (by default),
- Experiment identifier `deneme`, which will be used as the folder name in `results/` to save the experiment data,
- Saint Arnoult network and demand, from `networks/saint_arnoult`,
- Environment (also used for `random` and `numpy`) and PyTorch seeds 42 and 0, respectively.

```bash
python scripts/iql.py --id deneme --alg-conf config1 --task-conf config4 --net saint_arnoult --env-seed 42 --torch-seed 0
```

> Some scripts are compatible with only some certain task configurations. For example, scripts with "open" in the name work only with task configurations with "dynamic" in the name. 

####  Usage **URB** for baselines

Similarly as for RL algorithms, you have to provide command, but there is one additional flag ```model``` for ```scripts/baselines.py```, instead of ```torch-seed```, then you have command of form:

```bash
python scripts/baselines.py --id <exp_id> --alg-conf <hyperparam_id> --env-conf <env_conf_id> --task-conf <task_id> --net <net_name> --env-seed <env_seed> --model <model_name>
```

And ```<model_name>``` should be one of ```random```, ```aon``` (included in [baseline_models](baseline_models/)) or ```gawron``` (from [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/993423d101f39ea67a1f7373e6856af95a0602d4/routerl/human_learning/learning_model.py#L42)). 

For example:

```bash
python scripts/baselines.py --id ing_aon --alg-conf config1 --task-conf config2 --net ingolstadt_custom --model aon
```

## 📊 Calculating metrics and indicators  

Each experiment outputs set of raw records, which are then processed with the script in this folder for a set of performance indicators which we report and several additional metrics that track the quality of the solution and its impact to the system.

#### Usage

To use the analysis script, you have to provide in the command line the following command:

```bash
python analysis/metrics.py --id <exp_id> --verbose <verbose> --results-folder <results-folder> --skip-clearing <skip-clearing> --skip-collecting <skip-collecting>
```

that will collect the results from the experiment with identifier ```<exp_id>``` and save them in the folder ```<exp_id>/metrics/```. The ```--verbose``` flag is optional and if set to ```True``` will print additional information about the analysis process. Flag ```--results-folder``` is optional and if set to ```True``` will use the folder ```<results-folder>``` instead of the default one ```results/```. The flags ```--skip-clearing``` and ```--skip-collecting``` are optional and if set to ```True``` will skip clearing and collecting the results from the experiment, respectively. Those operations have to be done only once, so if you are running the analysis script multiple times, you can skip them.

#### Reported indicators

---

The core metric is the travel time $t$, which is both the core term of the utility for human drivers (rational utility maximizers) and of the CAVs reward.
We report the average travel time for the system $\hat{t}$, human drivers $\hat{t}\_{HDV}$, and autonomous vehicles $\hat{t}\_{CAV}$. We record each during the training, testing phase and for 50 days before CAVs are introduced to the system ( $\hat{t}^{train}, \hat{t}^{test}$, $\hat{t}^{pre}$). Using these values, we introduce: 

-  CAV advantage as  $\hat{t}\_{HDV}^{post}$ / $\hat{t}\_{CAV}$, 
-  Effect of changing to CAV as ${\hat{t}\_{HDV}^{pre}}/{\hat{t}\_{CAV}}$, and
-  Effect of remaining HDV as ${\hat{t}\_{HDV}^{pre}}/{\hat{t}\_{HDV}^{test}}$, which reflect the relative performance of HDVs and the CAV fleet from the point of view of individual agents.

To better understand the causes of the changes in travel time, we track the _Average speed_ and _Average mileage_ (directly extracted from SUMO). 

We measure the _Cost of training_, expressed as the average of: $\sum_{\tau \in train}(t^\tau_a - \hat{t}^{pre}_a)$ over all agents $a$, i.e. the cumulated disturbance that CAV cause during the training period. We define $c\_{CAV}$ and $c\_{HDV}$ accordingly.
We call an experiment _won_ by CAVs if their policy was on average faster than human drivers' behaviour. A final _winrate_ is a percentage of runs that were won by CAVs.
