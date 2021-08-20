clc
clear
weights=load("weights_200.mat");
weight=weights.weights;
w=weight{2};
[H,W,M,D]=size(w);
N=H*W*D;
cl=0;
for k=1:D
    for i=1:H
        for j=1:W
            cl=cl+(w(i,j,k)*(1-w(i,j,k)))/N;
        end
    end
end