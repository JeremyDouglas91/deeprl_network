# Networked Multi-agent RL (NMARL)

[NOTE]: this README has been updated for the purpose of running NMARL baseline experiments for the Sequantial Social Dilemma tasks Cleanup and Harvest.

Available IA2C algorithms:
* PolicyInferring: [Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." Advances in Neural Information Processing Systems, 2017.](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
* FingerPrint: [Foerster, Jakob, et al. "Stabilising experience replay for deep multi-agent reinforcement learning." arXiv preprint arXiv:1702.08887, 2017.](https://arxiv.org/pdf/1702.08887.pdf)
* ConsensusUpdate: [Zhang, Kaiqing, et al. "Fully decentralized multi-agent reinforcement learning with networked agents." arXiv preprint arXiv:1802.08757, 2018.](https://arxiv.org/pdf/1802.08757.pdf)


Available MA2C algorithms:
* DIAL: [Foerster, Jakob, et al. "Learning to communicate with deep multi-agent reinforcement learning." Advances in Neural Information Processing Systems. 2016.](http://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning.pdf)
* CommNet: [Sukhbaatar, Sainbayar, et al. "Learning multiagent communication with backpropagation." Advances in Neural Information Processing Systems, 2016.](https://arxiv.org/pdf/1605.07736.pdf)
* NeurComm: Inspired from [Gilmer, Justin, et al. "Neural message passing for quantum chemistry." arXiv preprint arXiv:1704.01212, 2017.](https://arxiv.org/pdf/1704.01212.pdf)

Available NMARL-SSD scenarios:
* Cleanup 
* Hervest

For more details see https://github.com/JeremyDouglas91/sequential_social_dilemma_games and https://arxiv.org/pdf/1810.08647.pdf.

## Usages

Available tasks:
* ia2c_cleanup
* ia2c_harvest
* ia2c_fp_cleanup
* ia2c_fp_harvest
* ma2c_cu_cleanup
* ma2c_cu_harvest
* ma2c_ic3_cleanup
* ma2c_ic3_harvest
* ma2c_dial_cleanup
* ma2c_dial_harvest
* ma2c_nc_cleanup
* ma2c_nc_harvest

For details on the hyperparameters in the cofig files see:

https://docs.google.com/spreadsheets/d/1RABudTdKMeUmkxPfKCASuT9bmSBcFkJPG--L1XLLj60/edit?usp=sharing

1. To train a new agent: 

Pull the docker image (if you havent already):
~~~
docker pull instadeepct/baselines:latest
~~~

Spin up the container (for example):

~~~
docker run -it --rm --gpus all -p 8888:8888 -p 6006:6006  -v "$(pwd)":/wd -w /wd --name ssd-baselines instadeepct/baselines:latest bash
~~~

Navigate to the `/tmp/deeprl_network` directory in the container, ensure the output folder (`/tmp/deeprl_network/output/`) is empty as there may be data from previous runs. Then, select a task from the list above and run:
~~~
python main.py --base-dir 'output/[task_name]' train --config-dir 'config/config_[task_name].ini
~~~

Training config/data and the trained model will be output to `output/[task_name]/data` and `output/[task_name]/model`, respectively.

2. To access tensorboard during training, run
~~~
tensorboard --logdir=output/
~~~

View the output in your browser at `localhost:[port]` where the port will either be 6006 or 8888 (as per the `docker run` command).

3. To evaluate a trained agent, run
~~~
python main.py --base-dir 'output/[task_name]' evaluate --evaluation-seeds [seeds]
~~~
Evaluation data will be output to `'output/[task_name]/eva_data`.     

[From the original authors:]

## Citation
For more implementation details and underlying reasonings, please check our paper [Multi-agent Reinforcement Learning for Networked System Control](https://openreview.net/forum?id=Syx7A3NFvH).
~~~
@inproceedings{
chu2020multiagent,
title={Multi-agent Reinforcement Learning for Networked System Control},
author={Tianshu Chu and Sandeep Chinchali and Sachin Katti},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Syx7A3NFvH}
}
~~~


