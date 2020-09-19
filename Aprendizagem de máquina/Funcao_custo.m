function [J] = Funcao_custo(X,preco,theta,regularizacao)
  m = length(preco);
  J = sum (( X * theta - preco).^2)./(2*m) + regularizacao;
end