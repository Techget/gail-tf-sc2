# Generative Adversarial Imitation Learning in tensorflow on PySC2
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

To get an idea of how I parse the .SC2Replay files, refer to [[parse recording file](https://github.com/Techget/parse-pysc2-replay-files)]

The trained parameter network should be put under `param_pre_model`. The pre-trained model is trained by running the codes in [[parameter model](https://github.com/Techget/pysc2-pretrained-parameter-model)], this pretrained model is used to supply the parameters for each 

In `master` branch, run `python3 main.py` to start training, the model will be saved every 100 episode

To evaluate, `git checkout UsePPOParameterSharingEvaluate` to evaluate the model, the trained model should be put in `/checkpoint`.

## Result
The result is not quite ideal, the agents only learns to construct building and have a few minions patrol around.

## Reference
## To inspect this project in detail, proceed to the *[[report](https://docs.google.com/document/d/16ceZp-Zdx4vxGpHDNZZ_dTbt1vbKYh0znUyDm5QpEfc/edit?usp=sharing)]* 
- Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning, [[arxiv](https://arxiv.org/abs/1606.03476)]
- @openai/imitation
- @openai/baselines

## Feel free to contact me if you want the trained model for the pretrained parameter network and GAIL network
