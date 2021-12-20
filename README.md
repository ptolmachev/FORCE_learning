# learning_with_FORCE
 FORCE learning algorithm from "*Generating Coherent Patterns of Activity from Chaotic Neural Networks*" (2009) by David Sussillo et al. 
 for training redout weights of the reservoir with the following dynamics:
 
https://render.githubusercontent.com/render/math?math=\tau \frac{d \bold{x}}{dt} = (W_{rec})f(\bold{x}) + w_{inp} I_{inp} + u_{fb} z.

where 

$$z = w_{out} f(\bold{x})$$

$f(x) = tanh(x)$
 
 Here is what it can do:
 
 ## Periodic output given periodic input:
![periodic output given periodic input](https://github.com/ptolmachev/FORCE_learning/blob/main/imgs/sine_wave_testing.png)

## Periodic output with zero input
![periodic output with zero input](https://github.com/ptolmachev/FORCE_learning/blob/main/imgs/triangle_wave_testing.png)

## Generating Mackey-Glass dynamics with no input
![generating Mackey-Glass dynamics with no input](https://github.com/ptolmachev/FORCE_learning/blob/main/imgs/mackey_glass_testing.png)

## Prediction task of Mackey-Glass dynamics
![Prediction task of Mackey-Glass dynamics](https://github.com/ptolmachev/FORCE_learning/blob/main/imgs/shifted_mackey_glass_testing.png)

## Delayed pulse response 
![Delayed pulse response](https://github.com/ptolmachev/FORCE_learning/blob/main/imgs/delayed_pulse_testing.png)
