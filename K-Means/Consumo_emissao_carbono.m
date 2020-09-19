% Trabalho de Marchine Learning, K-Means
% Autora: Lauany Reis da Silva
clear;clc;
% Pacote necessário para o funcionamento do Elbow Method
pkg load statistics;

X = load('data_1.mat'); 
% Separação dos dados para facilitar na manipulação
carbon = X.carbon;
millage = X.millage;
model = X.model;
% Emissão total de carbono por carro
Gasto_final = carbon .* millage;

% Para a escolha do K ótimo optou-se por usar 
% o Elbow Method, sendo que foi utilizada a implementação
% pronta no site do Matlab, sob licença Copyright (c) 2018, Sebastien De Landtsheer
% https://www.mathworks.com/matlabcentral/fileexchange/65823-kmeans_opt-optimal-k-means-with-elbow-method
[IDX,C,SUMD,K]=kmeans_opt(Gasto_final);
max_iterations = 10;

% Inicialização dos centroids com valores aleatórios
centroids = zeros(K,size(Gasto_final,2));
randidx = randperm(size(Gasto_final,1));
centroids = Gasto_final(randidx(1:K), :);

subplot(2,1,1);
hold on
axis([0 100 19 30])
title('Dados iniciais sem categorizacao');
ylabel('indice de emissao de carbono');
xlabel('Posicao no vetor');
grid minor;
plot(Gasto_final(:), 'm*');
plot(centroids(:),'xk');

for i = 1:max_iterations
    indices = getClosestCentroids(Gasto_final,centroids);
    centroids = computeCentroids(Gasto_final,indices,K);
end

subplot(2,1,2);
hold on
axis([0 100 19 30])
title('Dados categorizados. Rotular dados com o mouse: right exibe e left termina a visualizacao');
ylabel('indice de emissao de carbono');
xlabel('Posicao no vetor');
grid minor;
concepts = {};
plot(centroids(:),'xk');
% Usa o centroids_aux para categorizar de forma eficaz
centroids_aux = sort(centroids);
for i = 1:max_iterations
    indices = getClosestCentroids(Gasto_final,centroids_aux);
end
% Rotulação com cor e conceito de acordo com os centroids no eixo Y
for i = 1:80
      [color, concept] = lettering (indices (i));
      concepts(:,i) = concept;
      plot(i, Gasto_final(i:i), color);
end
% Legenda de cada categoria 
text (85, 29.5, "Pessimo", "color",'r');
text (85, 29, "Ruim", "color",'y');
text (85, 28.5, "Regular", "color",'c');
text (85, 28, "Bom", "color",'b');
text (85, 27.5, "Otimo", "color",'g');
% Médias de consumo solicitadas
targets ={"Pessimo", "Ruim", "Regular", "Bom", "Otimo"} ;
avg_m = [];
avg_c = [];
axi_y = 20;
text (85, axi_y, "Med consumo (classe)");
text (94, axi_y, "Med emissao (classe)");
for i = 1:5
    [average_millage, average_carbon] = consumptEmission (concepts, carbon, millage, targets(1,i));
    avg_m (i) = average_millage;
    avg_c (i) = average_carbon;
    axi_y += 1;
    text (85, axi_y, targets(1,i));
    text (94, axi_y, targets(1,i));
    axi_y -= 0.5;
    text (85, axi_y, mat2str(avg_m (i)));
    text (94, axi_y, mat2str(avg_c (i)));
    axi_y += 0.5;
end
hold off
% Parte interativa para a rotulação de cada carro pelo eixo X
but = 1;
while but == 1
    [xi, yi, but] = ginput(1);
    axe_x = round(xi);
    text(xi, yi,model(axe_x:axe_x));
    yi = yi - 1;
    text(xi, yi,concepts(round(xi)));
end

