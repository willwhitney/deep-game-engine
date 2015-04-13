# Things to try

## Reconstruction
- more feature maps on the decoder than encoder
    + especially in the earlier layers of decoder (widen the narrow end of the funnel)


## Prediction

### End to end
- reparametrize after prediction section
- reparametrizing twice
    + loss function is Loss[image_t+1, decoder(z_hat_t+1)] + KL(predictor) + KL(encoder)


### Fixed encoder/decoder
- predicting code layer z_hat_t+1, then testing reconstruction
    + Loss[image_t+1, decoder(z_hat_t+1)]
- predicting code layer, then testing against code layer
    + Loss[z_t+1, z_hat_t+1]

### Recurrence

