function mean_err = grad_check(f, n_inputs, n_checks, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% numerical gradient checking
% Inputs:
%   - f: is a function handle that returns [f_x, df_dx]:
%       f_x:   R^n -> R
%       df_dx: R^n -> R^n. calculates the derivaive with respect each 
%              dimention of x, in a point x
%   - n_inputs: size of the input vector for function f
%   - n_checks: number of checks
%   - varargin: additional inputs to function f
%
% Outputs:
%   - mean_error
%
% Created by: Daniel L. Marino (marinodl@vcu.edu)
% Modern Heuristics Research Group (MHRG) 
% Virginia Commonwealth University (VCU), Richmond, VA 
% http://www.people.vcu.edu/~mmanic/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol= 1e-7;

fprintf(' Iter |       err       |       num       |       given       \n')
fprintf(' ------------------------------------------------------------ \n')

dim= 0;
mean_err = 0;

n_out=nargout(f);
if n_out==-1
    n_out = 2;
end
for i=1:n_checks    
    x=rand(n_inputs, 1);
    %% pick two points to numerically differentiate
    dim= mod(dim, size(x, 1)) + 1; % pick a dimention to test
    
    x1= x;
    x1(dim)= x1(dim)-tol;
    
    x2= x;
    x2(dim)= x2(dim)+tol;
    
    %% numerically differentiate on dimention rand_d
    if n_out==2 
        f_x1= feval(f, x1, varargin{:});
        f_x2= feval(f, x2, varargin{:});
    elseif n_out==4
        [~,f_x1]= feval(f, x1, varargin{:});
        [~,f_x2]= feval(f, x2, varargin{:});
    end
    
    num_df= (f_x2 - f_x1)/(2*tol);
    
    %% get derivative given by df_dx
    %[~, giv_df]= feval(f, x, varargin{:});
    out = cell(n_out, 1);
    [out{:}]= feval(f, x, varargin{:});
    if n_out==2 
        giv_df = out{2};
    elseif n_out==4
        giv_df = out{4};
    end
    giv_df= giv_df(dim);
    
    %% calculate error
    err = abs( giv_df - num_df );
    mean_err = mean_err + err;
    
    fprintf('%5d | %15f | %15f | %15f | \n', i, err, num_df, giv_df)
end

mean_err = mean_err/n_checks;