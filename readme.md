# Deep networks for game prediction

This repo contains the code for a deep neural network which learns to predict the next frame of an Atari game from the previous one or two frames.

## Prediction results

### Single-frame prediction

**Prediction using a single frame as input.** The model takes in one frame of the game (the current screen) and produces a prediction for the next frame. The predicted frame for `t+1` is shown on the left in each column; the true frame which occurred at `t+1` is shown on the right side of each column.

Note the uncertainty about moving components.

<img src="slides/pred_1frame.png" width=500>

### Two-frame prediction

**Prediction using two frames as input.** The model takes in the two previous frames (`t-1` and `t`) and produces a prediction for the next frame `t+1`. The predicted frame for `t+1` is shown on the left in each column; the true frame which occurred at `t+1` is shown on the right side of each column.

Confidence about the future positions of moving objects is greatly improved versus the single-frame case.

<img src="slides/pred_2frame.png" width=500>





## Model

### Feature extraction with variational autoencoder

The code for the feature extraction is based on my NIPS paper from 2015: [https://github.com/willwhitney/dc-ign](https://github.com/willwhitney/dc-ign). This variational model yields a generative representation of the state.

![](slides/slides.002.png)
This diagram is for the DC-IGN; this model does not have the structure show in th hidden representation.

![](slides/slides.003.png)
![](slides/slides.004.png)

To ensure that even small moving components are captured by the autoencoder, I put a multiplier on the cost function for pixels that change from one frame to the next. This is extremely effective.

### Stable prediction in feature-space

![](slides/slides.005.png)
![](slides/slides.006.png)
![](slides/slides.007.png)

