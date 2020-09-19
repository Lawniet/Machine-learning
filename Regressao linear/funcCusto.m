function [J] = funcCusto(lambda,x,Y,theta)
  m = length(Y);
  J = sum((x*theta-Y).^2)./(2*m) + lambda*sum(theta.^2);
%  for i = 1:m
%    for j = 1:m
%      J(i,1) += pow2(H(i,1) - Y(j,1));
%    end
%    J(i,1)/(2*m);
%  end
end