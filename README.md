# RL_Project
The aim of this project is to test different Reinforcement Learning algorithms to create a self-driving-car using TORCS. We used the code of the python wrapper made by [ugo-nama-kun] (https://github.com/ugo-nama-kun/gym_torcs)

## Getting Started
### Requirements
We are using Ubuntu 64-bit 16.04.4 :
- Python 3
- gym_torcs (https://github.com/ugo-nama-kun/gym_torcs)
- OpenAI-Gym (https://github.com/openai/gym)
- Numpy

For further requirements refer to gym_torcs guide

### Installation
Follow the instruction explained on gym_torcs, but pay attention at two points:
1. Clone the gym_torcs directory into  `usr/src `, otherwise linking problems could come up
2. Use these command to compile vtorcs: 
``` 
sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng12-dev 
./configure --prefix=$(pwd)/BUILD  # local install dir
make
make install
make datainstall
./torcs 
```
as shown in (https://github.com/giuse/vtorcs/tree/nosegfault)

### Launching
To run the agent use : ``` sudo python3 example_experiment.py ``` 

The `example_experiment.py sample_agent.py ` are the class modified for this project

To check the results look at the J plot at the end of the process
