function [weights] = regularizedregression(train_data_X, train_data_Y, c, lambda)

%% TODO

%2 or 3 lines of code
Cd = diag(c,0);
dim = size(train_data_X,2);
weights = inv(train_data_X'*Cd*train_data_X + lambda*eye(dim))*train_data_X'*Cd*train_data_Y;
return;
