function [theta,J_it] = Gradiente_descendente(X,preco,thetaI,alpha,it)
  theta = thetaI;
  J_it = zeros(it,1);
  m = length(preco);
  lambda = 0.12;
  
  for i = 1: it
    temp0 = theta(1)-(alpha./m).*sum(X*theta-preco);
    temp1 = theta(2)-(alpha./m).*sum((X*theta-preco).*X(:,2));
    temp2 = theta(3)-(alpha./m).*sum((X*theta-preco).*X(:,2));

    theta(1) = temp0;
    theta(2) = temp1;
    theta(3) = temp2;
 
    regularizacao = lambda*sum(theta.^2);
    
    J_it(i) = Funcao_custo(X,preco,theta,regularizacao); 
  end
end

