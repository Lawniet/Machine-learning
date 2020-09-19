function [theta,J_it] = gradientDescedent(x,Y,thetaI,alpha,it)
  theta = thetaI;
  J_it = zeros(it,1);
  m = length(Y);
  for i = 1:it
    temp1 = theta(1) - (alpha./m).*sum(x*theta-Y);
    temp2 = theta(2) - (alpha./m).*sum((x*theta-Y).*x(:,2));
    theta(1) = temp1;
    theta(2) = temp2;
    J_it(i) = funcCusto(x,Y,theta);
  end
end