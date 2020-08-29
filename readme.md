# Neural Network Designer
## Introduction
Web application has been created for young developers to help them optimize or directly find optimal neural network according to
 their problem. Since almost everything can be stated as mathematical evidence( function) therefore possibilities of this application are unlimited.
 So let's suppose our company have some problem. Firstly we need to figure out what we want to maximalize, minimalize and want does not touching our interest.
 When we already found out that, we can state our problem as a mathematical evidence for example 
 `f(x)=x^2+5*x-73`. 
 Now we want to find a neural network 
 prototype that will be best(good enough) to solve our problem.
 When we are designing neural network model we need to select quantity of layer, neurons ( on each layer), activation functions and much more of things like these. 
  To sum up application for stated problem can choose optimal activations, layers number, neurons quantity  etc. It can save a lot of developer time and save some time to play a chess round or spend time with family.

Every model that is being created or improved for given function is stored using postgres in database and assigned to it's creator ;).
Each user before even contemplating on his own neural network could try to search through database for already found solutions to similar problems.
Web application using django user- login and registration system.

## How it works?
 Application is created to design neural networks what is a part of artificial intelligence.
  This program were developed to help and improve AI results using computational intelligence, exactly metaheuristic algorithms.
 To improve or even create new networks I have used algorithm- simulated annealing( SA). But how does it really work ?
 So it is an algorithm based on initial solution that in given time trying to improve it, by mutating model structure.
  If we already have some model that is good, but results are not enough for us, we can input that model as initial and 
  program will improve it otherwise the initial model will be created randomly. So during runtime SA has a parameter- 
  temperature that commands how far or where our current or next research mutation// acceptance should went. The problem in metaheuristics is
   that sometimes we can stuck in local optimum. SA handling that by moving to the worse solutions in accordance with
   temperature. After given time for example 30 seconds. Program return the most optimal solution that he could find on this
   time interval. [More Wikipedia SA](https://en.wikipedia.org/wiki/Simulated_annealing)
#### Initial Model
Can be alternatively input by:
1. model passed by user
2. Randomly created model
#### Mutation
Structure in next iterations can be mutated or disturbed by :
1. random layer deletion
2. random layer insertion
3. change activation on random layer
4. change neurons quantity on random layer
#### Local Optimum Problem Solution
1. Restarts after N iterations
2. Moving to worse solution in accordance with temperature ( initial T0 )

## Technologies

- Python 3.8
- Django 3.0
- Javascript
- PosgreSQL
- HTML 5
- CSS 3


- Tensorflow (GPU)
- Keras
- Sklearn


- Numpy
- Matplotlib
- Graphviz

# Requirments 
- pydotplus
- graphviz 
- tensorflow-gpu
- libcuda.so
