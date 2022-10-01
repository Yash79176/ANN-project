%% this is a program to implement artificial neural network

clc,clear all;

load('smaw.txt');                        %% loading of the dataset
L = input('Enter the number of inputs (MAX 10) ');
N = 1;                                                   %% the number of outputs         
P = input('Enter the number of training patterns (TOTAl No. OF PATTERNS 1000)');
Tp = input('Enter the number of testing patterns (TOTAl No. OF PATTERNS 1000)');
H = 10;                                                  %% the number of hidden neurons
et = 0.85;                                               %% learning rate
a=0.75;                                                  %% momentum rate

data1=smaw;

random1 = randi([1,size(smaw,1)],1,size(smaw,1));        %% radomizing the input dataset
for i=1:length(random1)
    data1(i,:) = smaw(random1(i),:);
end

T1 = data1(1:P,end-N+1:end );                            %% target values to find error
m = max(T1);
n = min(T1);

maximum = max(data1,[],1);                               %% normalization of dataset
minimum = min(data1,[],1);
[r , c ] = size(data1);
for i=1:r
    for j=1:c
        data1(i,j) = 0.1 + 0.8*( (data1(i,j) - minimum(j))/(maximum(j)-minimum(j)) );
    end
end

I = ones(L+1,P);                                         %% input for training set

I(2:L+1,1:P) = data1(1:P,1:L)';

It = ones(L+1,Tp);                                       %% input for test set

It(2:L+1,1:Tp) = data1(P+1:P+Tp,1:L)';

%%for H=1:20                                             %% for loop for diff. hiden layer

V = ones(L+1,H);

W = ones(H+1,N);

for i=2:L+1                                              %% random intialization of weights
    for j=1:H
        V(i,j) = random('Normal',0,0.333);
    end
end

for i=2:H+1
    for j=1:N
        W(i,j) = random('Normal',0,0.333);
    end
end

 T = data1(1:P,end-N+1:end );               
Ts = data1(P+1:P+Tp,end-N+1:end );

delV_t = zeros(L+1,H);
delW_t = zeros(H+1,N);

no_it = 1;
MSE =1;

while no_it < 10000
    
    Oo = zeros(N,P);
    Ih = zeros(H,P);
    Oh = zeros(H,P);
    Io = zeros(N,P);
    
    %% forward propagation
    for p=1:P
       for j=1:H 
           for i=1:L+1
               Ih(j,p) = Ih(j,p)+ I(i,p)*V(i,j);
           end
       end
    end
    
    Oh = 1 + exp(-1*Ih);
    Oh = 1./Oh;
    Oh = [ ones(1,P) ; Oh ];
    
    for p=1:P
       for j=1:N 
           for i=1:H+1
               Io(j,p) = Io(j,p) + Oh(i,p)*W(i,j);
           end
       end
    end
   
    Oo = 1 + exp(-1*Io);
    Oo = 1./Oo;
    Oo = Oo';
    
    E = (T - Oo).^2;
    Er = 0.5*E;
    Er = sum(Er,2);
    MSE = sum(Er)/P;                                    %% error calculation
    
    %% back propagation of error calculated
    delV = zeros(L+1,H);
    delW = zeros(H+1,N);
    
    for i=1:H+1
        for j=1:N
            for p=1:P
               delW(i,j) = delW(i,j) + ( (T(p,j)-Oo(p,j))*Oo(p,j)*(1-Oo(p,j))*Oh(i,p) );
            end
            delW(i,j) = delW(i,j)*(et/P);
        end
    end

   
    for i=1:L+1
        for j=1:H
            for p=1:P
                for k=1:N
                    delV(i,j) = delV(i,j) + ( (T(p,k)-Oo(p,k))*Oo(p,k)*(1-Oo(p,k))*W(j+1,k)*Oh(j+1,p)*(1-Oh(j+1,p))*I(i,p) );
                end
                delV(i,j) = delV(i,j)*(1/N);
            end
            delV(i,j) = delV(i,j)*(1/P)*(et);
        end
    end
    
    %% weight updation
    W = W + delW + a*delW_t;
    V = V + delV + a*delV_t;
    
    delV_t = delV;
    delW_t = delW;
    
    NMSE(no_it) = MSE;
  
    no_it = no_it + 1;
    
end                                       %% End of while loop
 
 %%MSE_arr(H) = MSE;
%%end                                     %% End of for loop for hidden layer 
  
%% denormalization of output
  for p=1:P
        for q=1:N
            Oo(p,q) = ((Oo(p,q)-0.1)*(m(q)-n(q))/0.8) + n(q);
        end
  end
   
  %% caluation of MSE of testset
    Oot = zeros(N,Tp);
    Ih = zeros(H,Tp);
    Oh = zeros(H,Tp);
    Io = zeros(N,Tp);

    for p=1:Tp
       for j=1:H 
           for i=1:L+1
               Ih(j,p) = Ih(j,p)+ It(i,p)*V(i,j);
           end
       end
    end
    
    Ih = [ ones(1,Tp) ; Ih ];
    Oh = 1 + exp(-1*Ih);
    Oh = 1./Oh;
    
    for p=1:Tp
       for j=1:N 
           for i=1:H+1
               Io(j,p) = Io(j,p) + Oh(i,p)*W(i,j);
           end
       end
    end
    
    Oot = 1 + exp(-1*Io);
    Oot = 1./Oot;
    Oot = Oot';
    
    tE = (Ts - Oot).^2;
    tEr = 0.5*tE;
    tEr = sum(tEr,2);
    T_MSE = sum(tEr)/Tp;
    
  
  %% writing the results to output file
fi = fopen('OUTPUT','w');
fprintf(fi,'\nTotal number of iterations is %d\n',no_it);
fprintf(fi,'MEAN SQUARE ERROR ( in training set ) - %f\n',MSE);
fprintf(fi,'MEAN SQUARE ERROR for the test set (ERROR IN PREDICTION)  - %f\n',T_MSE);
fprintf(fi,'\nThe output of the Neural network is \n');

    for i=1:P
        for j=1:N
            fprintf(fi,'%d\n',Oo(i,j));
        end
    end

    
    Error = abs(T1 - Oo);
    Eff = (T1 - Oo).^2;
    Erff = 0.5*Eff;
    Erff = sum(Erff,2);
    MSE_final = sum(Erff)/P;
    
    fprintf(fi,'\n The absolute error of the Neural network in prediction is \n');

    for i=1:P
        for j=1:N
            fprintf(fi,'%d\n',Error(i,j));
        end
    end


%% plotting
x = [1:(no_it-1)];                              %% plot of iterations vs MSE
plot(x,NMSE,'m:s');
xlabel('ITERATION');
ylabel('MSE');

%%Y = [1:20];                                   %%  plot of no. hidden neurons vs
%%plot(Y,MSE_arr,'g:s');                       
%%xlabel('NO. OF HIDDEN NEURONS');
%%ylabel('MSE');

    
    fprintf("THE OUTPUT IS STORED TO FILE \n");