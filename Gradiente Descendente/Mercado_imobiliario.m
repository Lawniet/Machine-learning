clear;
clc;

% 1ª coluna = área do imóvel (em pés quadrados "squared feet")
% 2ª coluna = número de quartos 
% 3ª coluna = valor desse imóvel
Imovel = load("data.mat");
         
area = Imovel(1:47,1);
qtd_quartos = Imovel(1:47,2);
preco = Imovel(1:47,3);

theta = [1;2;3]; 

l = length(area);
X = [ones(l,1) area qtd_quartos];
regularizacao = 0.0001;

J = Funcao_custo(X,preco,theta,regularizacao); 
[gd,J_it] = Gradiente_descendente(X,preco,theta,0.0000000007,1000);

subplot(2,2,1)
plot3(area,qtd_quartos,preco,"or");
title('Graficos do Mercado Imobiliario');
ylabel('Quantidade de quartos');
xlabel('Area');
zlabel('Preco');
legend('Mercado Imobiliario');
grid on;

hold on;
subplot(2,2,2)
plot(preco,'rx');
legend('Distribuicao de precos');
subplot(2,2,3)
plot(gd,'b-');
legend('Previsao de precos');
subplot(2,2,4)
plot(J_it,'b-');
legend('Deslocamento de precos');
grid on;
hold off;

Preco_estimado = gd(1) + gd(2)*1650 + gd(3) * 3;
Preco_estimado

save('registro.mat');
