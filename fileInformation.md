# INSTALLATION

It is necessary to have a Linux system to run the system and the instructions are specific to Ubuntu 24.04 LTS system. Other Linux systems may also work but your setup may need to be fine-tuned to your specific version.

Set the directory where project is installed. Eg. Download the `dev` directory as `~/runtime` so that contents of `dev` directory appear under `~/runtime` and point `INSTALL_PATH` bash environment variable to it.

```bash
export INSTALL_PATH=$HOME/runtime
```

Ensure that micromamba is installed on the system and a micromamba environment named `rl` is defined with the Python packages preinstalled. A `util/micromamba-rlenv.txt` is provided to duplicate a sample run environment.

Micromamba installation typically puts micromamba specific environment setup in `.bashrc`. However, a separate `util/.activate_umamba.sh` has been provided here that may be copied to `~/.activate_umamba.sh` so that you can explicitly source it in your terminal or scripts.

```bash
source  ~/.activate_umamba.sh
micromamba create -n rl python=3.12 pip
micromamba run -n rl python -m pip install -r util/micromamba-rlenv.txt
micromamba activate rl
cp $INSTALL_PATH/util/.activate_umamba.sh ~/.activate_umamba.sh
```

Activate the micromamba environment using the following command and use it for all terminals:

```bash
source  ~/.activate_umamba.sh
micromamba activate rl
```

Additionally, external simulators are needed to run the system so that the system interacts with this environment to perform useful work. A separate simulator is needed to test each domain and there may be more than one simulator available for each test domain.

Since simulators are very specific to the test domain and necessary environments, often involving many external packages and tools, it is generally much more involved to setup each simulator. Please refer to any installation guideline available for a simulator and ensure you follow all domain and simulator specific constraints to build your configuration file.

This must be repeated across all systems that are to run one or more entities that are part of an experiment. It is best to have the software in the same directory structure on all systems under the same username on each target machine.

Additionally, pick machines which have good GPU support for CUDA as a lot of heavy runtime processing gets done very quickly on systems with good GPU support. It is highly discouraged to run experiments on systems without proper GPU support.

# RUNNING EXPERIMENTS

To run the experiments, install the necessary components as described in the installation section.

Assuming that the system is installed at path `INSTALL_PATH=~/runtime`, run the system as follows:

```bash
source ~/.activate_umamba.sh
micromamba activate rl
cd $INSTALL_PATH
export PYTHONPATH="$PWD/src"
```

Create a configuration file as needed for your experiments. Set it up for your target test domain and available systems in your test environment.

```bash
export CONFIG_FILE=...   # E.g. $PWD/config/runtime.example.json
```

Setup passwordless SSH access for the participating hosts:

```bash
ssh-copy-id $USER@localhost
```

Repeat the above command for `$USER` on additional hosts participating in the experiment and again as needed.

Start the lifecycle launcher on the configured hosts so that they can instantiate configured entities for each machine.

```bash
python -m runtime.lifecycle_launcher --user $USER --basepath ${INSTALL_PATH} --config=${CONFIG_FILE}
```

The following command can be used to check if the lifecycle servers are up on each of the nodes:

```bash
ps -ef | grep host_lifecycle_server
```
It should list the expected lifecycle server(s) on the node. 

Launch your experiment-driving test-case implementations such as `test_harness`. A `--seed #` argument with # as some random number may be used to run for that seed value for test repeatability.

```bash
python -m runtime.test_harness --config ${CONFIG_FILE}
```

If the system is not setup correctly, missing required components such as the external-simulator, assignment-optimizer, risk-transformer, RL-Meta-policy provider, task models provider, or the policy-bank the system will throw an exception indicating the cause of error. A normally running system should produce some messages on command line indicating operational messages across the system entitites.
