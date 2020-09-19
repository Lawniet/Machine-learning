function [theta,J_it] = gradientDescedent(lambda,x,Y,thetaI,alpha,it)
  theta = thetaI;
  J_it = zeros(it,1);
  m = length(Y);
  
  #theta = theta - (alpha./m).*sum(x*theta-Y).*x.;
  
  for i = 1:it
    temp1 = theta(1) - (alpha./m).*sum(x*theta-Y) + lambda*sum(theta);
    temp2 = theta(2) - (alpha./m).*sum((x*theta-Y).*x(:,2)) + lambda*sum(theta);
    temp3 = theta(3) - (alpha./m).*sum((x*theta-Y).*x(:,3)) + lambda*sum(theta);
    theta(1) = temp1;
    theta(2) = temp2;
    theta(3) = temp3;
    J_it(i) = funcCusto(lambda,x,Y,theta);
  end
end