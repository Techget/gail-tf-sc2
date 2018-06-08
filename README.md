# Check out the simpler version at [openai/baselines/gail](https://github.com/openai/baselines/blob/master/baselines/gail/README.md)!
# gail-tf on PySC2
Tensorflow implementation of Generative Adversarial Imitation Learning, and apply GAIL on PySC2
**disclaimers**: some code is borrowed from @openai/baselines and @andrewliao

## What's GAIL?
- model free imtation learning -> low sample efficiency in training time 
  - model-based GAIL: End-to-End Differentiable Adversarial Imitation Learning
- Directly extract policy from demonstrations
- Remove the RL optimization from the inner loop od inverse RL
- Some work based on GAIL:
  - Inferring The Latent Structure of Human Decision-Making from Raw Visual 
    Inputs
  - Multi-Modal Imitation Learning from Unstructured Demonstrations using 
  Generative Adversarial Nets
  - Robust Imitation of Diverse Behaviors
  
## Requirements
- python==3.5.2
- mujoco-py==0.5.7
- tensorflow==1.1.0
- gym==0.9.3

## Run the code
Actions in PySC2 is composed of action id and extra parameters, eg to move a minion, RL agents need to provide corresponding action id and coordinates on map. I use GAIL to learn to choose reasonable action id, and use a separate supervised learning neural network to obtain correct parameters.

### Step 2: Imitation learning

#### Imitation learning via GAIL

```bash
python3 main.py --env_id $ENV_ID --expert_path $PICKLE_PATH
```

Usage:
```bash
--env_id:          The environment id
--num_cpu:         Number of CPU available during sampling
--expert_path:     The path to the pickle file generated in the [previous section]()
--traj_limitation: Limitation of the exerpt trajectories
--g_step:          Number of policy optimization steps in each iteration
--d_step:          Number of discriminator optimization steps in each iteration
--num_timesteps:   Number of timesteps to train (limit the number of timesteps to interact with environment)
```

To view the summary plots in TensorBoard, issue
```bash
tensorboard --logdir $GAILTF/log
```

##### Evaluate your GAIL agent
```bash
python3 main.py --env_id $ENV_ID --task evaluate --stochastic_policy $STOCHASTIC_POLICY --load_model_path $PATH_TO_CKPT --expert_path $PICKLE_PATH
```

## TroubleShooting

- encounter `error: Cannot compile MPI programs. Check your configuration!!!` or the systme complain about `mpi/h` 
```bash
sudo apt install libopenmpi-dev
```

## Reference
- Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning, [[arxiv](https://arxiv.org/abs/1606.03476)]
- @openai/imitation
- @openai/baselines
