clear
clc
close all

%ANN parameter
numSamp = 10000;
eta = 0.1;
numNeu = 1;

%Initialization
W = rand(numNeu,2);
V = rand(1,numNeu);
sumErr = 0;

%Training
index = [];
MSE = [];
for step = 1:numSamp
    
    %Feed forward path
    x = [;1];
    actFunc = W*x;
    decFunc = tanh(actFunc);
    y.hat = V*decFunc;
    
    %Back Propagation
    y.act = 2*x(1)^2+1;
    error = y.act - y.hat;
    V = V + eta*error*(decFunc)';
    w = W + eta*error*V' .* (1-decFunc) .* (1+decFunc) * x' ;
    
    sumErr = sumErr + 0.5*error^2;
    index(step) = step;
    MSE(step) = sumErr/step;
   
end