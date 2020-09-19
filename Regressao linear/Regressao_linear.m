clear;clc;

M = load("data_2.mat");
A = M(:,1);
N = M(:,2);
P = M(:,3);
lambda = 0.0001;
theta = [1;2;3];
l = length(A);
m = [ones(l,1) A N];

#[gd,J_it] = gradDesc3var(lambda,m,P,theta,0.0000000007,1000);
ID = eye(length(theta));
ID(1) = 0;
gd = pinv(m' * m + lambda.*ID) * (m'*P);

subplot(2,2,1)
plot3(A,N,P,"or");
subplot(2,2,2)
plot(A,1000*N,"or",m,m,"b");
subplot(2,2,3)
#plot(J_it,"r");
subplot(2,2,4)
plot(gd,"r");

gd(1) + gd(2)*1650 + gd(2)*3