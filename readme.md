
# Neural Network Designer
## Introduction
Web application has been created for young developers to help them optimize or directly find optimal neural network according to
 their problem. Since almost everything can be stated as mathematical evidence( function) we are able to use this problem-solving application.
 So let's suppose our company have some problem. Firstly we need to figure out what we want to maximalize, minimalize and want does not touching our interest.
 When we already found out that, we can state our problem as a mathematical evidence for example 
 `f(x)=x^2+5*x-73`. 
 Now we want to find a neural network 
 prototype that will be best(good enough) to solve our problem.
 When we are designing neural network model we need to select quantity of layer, neurons ( on each layer), activation functions and much more of things like these. 
 So just to simplify our lives this program is open- source :).
 Program for stated problem can choose optimal activations, layers number, neurons quantity  etc.
 If we already found some model that is good, but it is not enough result for us we can input that model as initial otherwise the initial model will be created randomly.
 
 
 
 
 mutation by :
- change neurons quantity on random layer
- change activation on random layer
- delete random layer

Requirments 
- pydotplus
- graphviz 
- tensorflow-gpu
- libcuda.so