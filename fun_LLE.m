%%  Locally Linear Embedding (LLE) 
% It is created by R. Hajizadeh PhD in electrical engineering
% LLE is a local Manifold Learning method that consists three steps.
function Y =fun_LLE(X,K,d)
% ## Inputs:
% X is the matrix of high-dimensional data points (D * N) where N in the number of the data points and D is the dimensionality of each data point.
% K in the number of the neighbours
% d is the dimensionality of the data points in low-dimensional representation space

% ## Output:
% Y is the matrix of data points in low-dimensional space.

[D,N] = size(X);
%% STEP1: Finding k neighbours of each data point
X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;
[sorted,index] = sort(distance);
neighborhood = index(2:(1+K),:);

%% STEP2: Calculating the reconstruction coefficients (W) and then neighbourhood graph matrix (M)
W = zeros(K,N);
for c1=1:N
    z = X(:,neighborhood(:,c1))-repmat(X(:,c1),1,K); % shift ith pt to origin
    C = z'*z;                                        % local covariance
    W(:,c1) = C\ones(K,1);                           % solve Cw=1
    W(:,c1) = W(:,c1)/sum(W(:,c1));                  % enforce sum(w)=1
end;

temp_W = zeros(N);
for c2 =1:N
    temp_W(c2,neighborhood(:,c2))=W(:,c2);
end
M = (eye(N)-temp_W)'*(eye(N)-temp_W);

%% STEP3: Calculating the embedded data points in the low-dimensional space
rank_M = rank(M);
temp_adad_M = N - rank_M;
[Y_M,eigenvals_M] = eig(full(M));
[sorted_eigValue_M, ind3_M]= sort(abs(diag(eigenvals_M)));
if rank_M < d
    Y = Y_M(:,ind3_M((end-d+1):(end)))'*sqrt(N);
else
    Y = Y_M(:,ind3_M((temp_adad_M+1):(temp_adad_M+d)))'*sqrt(N);
end
