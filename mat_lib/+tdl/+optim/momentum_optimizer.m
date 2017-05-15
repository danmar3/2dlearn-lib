function [xp, f_xp] = momentum_optimizer(f, x0, options, varargin)
% Gradient descent optimizer with momentum
% Inputs:
%   @param f: is a function handle that returns [f_x, df_dx]:
%       f_x:   R^n -> R
%       df_dx: R^n -> R^n. calculates the derivaive with respect each 
%              dimention of x, in a point x
%   @param x0: initial point
%   @param options: options for the optimizer, created with 
%              tdl.optim.optimoptions
%   @param varargin: additional inputs to function f
%
% Outputs:
%   @retval xp is the optimal point
%   @retval f_x value of f at point xp
%
% Wrote by: Daniel L. Marino (marinodl@vcu.edu)
%   Modern Heuristics Research Group (MHRG)
%   Virginia Commonwealth University (VCU), Richmond, VA
%   http://www.people.vcu.edu/~mmanic/

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

if options.isKey('Momentum')
    momentum = options('Momentum');    
else
    momentum = 0.9;
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

if options.isKey('MaxValue')
    MaxValue = options('MaxValue');
else
    MaxValue = Inf;
end


if options.isKey('MaxIterations')
    MaxIterations = options('MaxIterations');
else
    MaxIterations = Inf;
end

n_out=nargout(f);
if n_out==-1
    n_out = 2;
end

xp = x0; xs1= x0;
iter  = 0;

v = 0;
while true    
    % evaluate f(x), df(x)
    fout = cell(n_out, 1);
    [fout{:}]= feval(f, xp, varargin{:});
    if n_out==2
        f_xp = fout{1};
        df_dx = fout{2};
    end
    
    if f_xp > MaxValue
        fprintf('non-finite f(x) found. Skiping to next iteration \n')
        xp= xs2;        
        iter = iter+1;
        continue
    end
    
    if f_xp < 0.1*MaxValue
        xs2 = xs1;
        xs1 = xp;
    end
    % update decision variables
    v = momentum*v + lr(iter)*df_dx;
    xp = xp - v;
    
    % log information
    if mod(iter,log_iter)==0
        fprintf('%5d | %15f | %15f | \n', iter, f_xp, lr(iter))
    end
    
    % check end conditions
    if iter>MaxIterations
        break
    end
    iter = iter+1;   
end

