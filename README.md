# DAgger_example

tested on ubuntu 18.04LTS

# Installation
1. install torcs
    ```
    cd vtorcs-RL-color
    sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev ibplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev   libpng-dev
    ./configure
    sudo make
    sudo make install 
    sudo make datainstall
    ```
2. install xautomation
    for key automation
    ```
    sudo apt-get update -y
    sudo apt-get install -y xautomation
    ```

# Training Learner
first need to change configureation of game environment
```
# start torcs env
sudo torcs -vision
```
1. go to race configuration and select racing track and set the driver `scr_server 1`
    - `race -> practice -> configure race -> select racing track -> select driver`
2. go to display setup, select `64x64` RGB observation and `16` color depth
    - `options -> Display`

start training!
```
python train.py
```
# Structure
```
DAgger_example
├── README.md
├── train.py
│     train agent with DAgger algorithm
│
├── agent.py 
│     neural network architecture of learner
│
├── snakeoil3_gym.py
│     Torcs client communicate with Torcs game server
│ 
├── gym_torcs.py
│     make Torcs client gym environment
│
├── autostart.sh
│     key automation
│
├── vtorcs-RL-color
│    Torcs source, copied from https://github.com/ugo-nama-kun/gym_torcs
│
├── images
        sampled images
```