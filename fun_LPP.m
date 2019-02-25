%%  locality preserving projection (LPP)
% It is created by R. Hajizadeh (PhD in electrical engineering)
% LPP is a linear local Manifold Learning method that consists three steps.
% In LPP, a mapping matrix is achieved which can be used for representation of data points in the low-dimensional space.
function Y_LPP = fun_LPP(X,K,d,t)
% ## Inputs:
% X is the matrix of high-dimensional data points (D * N) where N in the number of the data points and D is the dimensionality of each data point.
% K in the number of the neighbours.
% d is the dimensionality of the data points in low-dimensional representation space.
% t is heat kernel or variance parameter.

% ## Output:
% Y_LPP is the matrix of data points in low-dimensional space.

[D,N] = size(X);
%% STEP1: Finding K neighbours of each data point
X2 = sum(X.^2,1);
euclidian_dist = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X; 
[sorted1,index1] = sort(euclidian_dist);
neighborhood_euclidian = index1(2:(1+K),:);

%% STEP2: Calculating the reconstruction coefficients (W) and then construction the neighbourhood graph matrix
PHi = zeros(N,N);
for k1=1:N
PHi(k1,neighborhood_euclidian(:,k1)) = exp(-euclidian_dist(k1,neighborhood_euclidian(:,k1))/(2*t^2));
%  PHi(k1,neighborhood_euclidian(:,k1)) = 1; 
end
PHi = max(PHi, PHi');

%% calculation of the mapping matrix 
D_LPP = diag(sum(PHi));
L_LPP = D_LPP - PHi;
temp_LPP = rank(((X*D_LPP*X')^-1)*X*L_LPP*X');
[LPP_Vectors , LPP_values] = eig(((X*D_LPP*X')^-1)*X*L_LPP*X'); % calcula
[diag_LPP_values , ind2] = sort(diag(LPP_values));
temp_LPP = find(diag_LPP_values<1e-7);
U_LPP = LPP_Vectors(:,ind2((length(temp_LPP)+1):(length(temp_LPP)+d)));

%% calculation of the embedded data points in the low-dimensional space using mapping matrix
Y_ LPP = U_LPP'*X;