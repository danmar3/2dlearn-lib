function [ y ] = batch_matmul( x1, x2, ind )
% batch_matmul matrix multiplication between two 3d tensors
%
%   This function was written to match tensorflow's batch_matmul function
%   The input tensors x1 and x2 are 3-D with shape [..., r_x, c_x] and
%   [..., r_y, c_y]. The matrix multiplication is performed assuming that
%   each element indexed by the 1-dim is a matrix
% 
% Created by: Daniel L. Marino (marinodl@vcu.edu)
% Modern Heuristics Research Group (MHRG) 
% Virginia Commonwealth University (VCU), Richmond, VA 
% http://www.people.vcu.edu/~mmanic/
    
%{
rank = max(ndims(x1), ndims(x2));

if (nargin == 2) && (rank==3)
    ind = [3, 2];
end

% transpose
perm_ind= 1:rank;
perm_ind(1) = ind(1); perm_ind(ind(1)) = 1;
perm_ind(2) = ind(2); perm_ind(ind(2)) = 2;


x1 = permute(x1, perm_ind);
x2 = permute(x2, perm_ind);

% perform batch multiplications
%}

n_mat = max(size(x1,1), size(x2,1));
n = size(x1,2);
m = size(x2,3);
% transpose the tensors in a way that it matches the column-major order
% used by matlab
x1= permute(x1, [2,3,1]);
x2= permute(x2, [2,3,1]);

y= zeros(n, m, n_mat);

if size(x1, 3)== 1
    for i= 1:n_mat
        y(:,:,i)= x1*x2(:,:,i);
    end
elseif size(x2, 3)== 1
    for i= 1:n_mat
        y(:,:,i)= x1(:,:,i)*x2;
    end
else
    for i= 1:n_mat
        y(:,:,i)= x1(:,:,i)*x2(:,:,i);
    end
end

% transpose y to follow desired output format
y= permute(y, [3,1,2]);


end

