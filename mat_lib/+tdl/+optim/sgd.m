function [x, f_x] = sgd(f, x0, options, varargin)
% Stochastic Gradient descent optimizer
% Inputs:
%   - f: is a function handle that returns [f_x, df_dx]:
%       f_x:   R^n -> R
%       df_dx: R^n -> R^n. calculates the derivaive with respect each 
%              dimention of x, in a point x
%   - x0: initial point
%   - options: options for the optimizer, created with 
%              tdl.optim.optimoptions
%   - varargin: additional inputs to function f
%
% Outputs:
%   @retval xp is the optimal point
%   @retval f_x value of f at point xp
%
% Wrote by: Daniel L. Marino (marinodl@vcu.edu)


tol= 1e-7;
header =strcat(' Iter |       f(x)      |      lr       |\n', ...
               '-----------------------------------------\n');
fprintf(header)

% check optim options
if options.isKey('LearningRate')
    if isa(options('LearningRate'),'function_handle')
        lr = options('LearningRate');
    elseif isa(options('LearningRate'),'double')
        lr = @(i) options('LearningRate');
    end
else
    lr = @(i) 1;
end

if options.isKey('Display')
    if strcmp(options('Display'), 'Iter')
        log_iter = 1;
    else
        log_iter = options('Display');
    end    
else
    log_iter = 1;
end

if options.isKey('MaxIterations')
    MaxIterations = options('MaxIterations');
else
    MaxIterations = Inf;
end

mean_err = 0;
n_out=nargout(f);
if n_out==-1
    n_out = 2;
end

x = x0;
iter  = 0;
while true    
    % evaluate f(x), df(x)
    fout = cell(n_out, 1);
    [fout{:}]= feval(f, x, varargin{:});
    if n_out==2
        f_x = fout{1};
        df_dx = fout{2};
    end
        
    % update decision variables
    x = x - lr(iter)*df_dx;
    
    % log information
    if mod(iter,log_iter)==0
        fprintf('%5d | %15f | %15f | \n', iter, f_x, lr(iter))
    end
    
    % check end conditions
    if iter>MaxIterations
        break
    end
    iter = iter+1;   
end

