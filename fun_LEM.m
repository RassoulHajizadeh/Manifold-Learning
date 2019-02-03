%%  Laplacian Eigen Map (LEM)
% It is created by R. Hajizadeh (PhD in electrical engineering)
% LEM is a local Manifold Learning method that consists three steps.
function Y_LEM = fun_LEM(X,K,d,t)
% ## Inputs:
% X is the matrix of high-dimensional data points (D * N) where N in the number of the data points and D is the dimensionality of each data point.
% K in the number of the neighbours.
% d is the dimensionality of the data points in low-dimensional representation space.
% t is heat kernel or variance parameter.

% ## Output:
% Y is the matrix of data points in low-dimensional space.

[D,N] = size(X);
%% STEP1: Finding k neighbours of each data point
X2 = sum(X.^2,1);
euclidian_dist = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X; % sum(||X-Y||^2)
[sorted1,index1] = sort(euclidian_dist);
neighborhood_euclidian = index1(2:(1+K),:);

%% STEP2: Calculating the reconstruction coefficients (W) and then construction the neighbourhood graph matrix
W = zeros(N,N);
for k1=1:N
    W(k1,neighborhood_euclidian(:,k1)) = exp(-euclidian_dist(k1,neighborhood_euclidian(:,k1))/(2*t^2)); 
end
W = max(W,W');

%% calculation of the embedded data points
D_LEM = diag(sum(W));
L_LEM = D_LEM - W;

[preY_LEM , values] = eig(D_LEM^-1*L_LEM);
[eigenvalues , ind1] = sort(diag(values));
temp_flag = find(eigenvalues<1e-7); % Or Rank of D_LEM^-1*L_LEM matrix can be calculated and used.

Y_LEM = preY_LEM(:,ind1((length(temp_flag)+1):(length(temp_flag)+d)));

