function [ y ] = batch_mat_diag( x )
% batch_mat_diat: creates a tensor with diagonal matrices indexed by the
%                 1-dim, with the diagonal equal to the corresponding row
%                 of x
%
%   This function was written to match tensorflow's batch_mat_diag function
%   The input tensor x must have shape [..., diag_elem].
% 
% Wrote by: Daniel L. Marino (marinodl@vcu.edu)
% Modern Heuristics Research Group (MHRG) 
% Virginia Commonwealth University (VCU), Richmond, VA 
% http://www.people.vcu.edu/~mmanic/

n_mat = size(x,1);
n = size(x,2);

y= zeros(n, n, n_mat);

for i= 1:n_mat
    y(:,:,i)= diag(x(i,:));
end

% transpose y to follow desired output format
y= permute(y, [3,1,2]);

end

