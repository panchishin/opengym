# Pole Cart 

Training an agent to balance a pole on a cart.


## Artificial Evolution

The agent 'learns' through artificial evolution.
```
python cartpole.py
```

<img src="./pics/smooth.gif" alt="A trained agent" width="400"/>

The script stores the best agent in `cartpole_best.pickle`.  Delete that file if you want to start over.

The script takes some time to get here.  Here is a video of it struggling in the beginning

<img src="./pics/struggle.gif" alt="An untrained agent" width="400"/>


## SARSA

cartpole_sarsa.py uses sarsa to learn how to control the cart.
```
python cartpole_sarsa.py
```
<img src="./pics/struggle.mp4" alt="Sarsa training" width="400"/>




I used ffmpeg to create the above gif
```
ffmpeg -i smooth.mov -s 400x200 -pix_fmt rgb8 -r 30 -f gif - | gifsicle --optimize=3 --delay=3 > smooth.gif
```

Other environments of interest

https://gym.openai.com/envs/BipedalWalker-v2/

https://gym.openai.com/envs/CarRacing-v0/

