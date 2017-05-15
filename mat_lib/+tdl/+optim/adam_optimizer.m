function [xp, f_xp] = adam_optimizer(f, x0, options, varargin)
% Adam optimizer implementation from the paper "Adam: A method for
%    stochastic optimization", Kingma, Diederik and Ba, Jimmy, arXiv
%    preprint arXiv:1412.6980 
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
if options.isKey('alpha')
    alpha = options('alpha');
else
    alpha = 0.001;
end

if options.isKey('beta_1')
    beta_1 = options('beta_1');
else
    beta_1 = 0.9;
end

if options.isKey('beta_2')
    beta_2 = options('beta_2');
else
    beta_2 = 0.999;
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

n_out=nargout(f);
if n_out==-1
    n_out = 2;
end

xp = x0; 
mt = 0;
vt = 0;
t  = 0;

while true    
    t = t + 1;
    % evaluate f(x), df(x)
    fout = cell(n_out, 1);
    [fout{:}]= feval(f, xp, varargin{:});
    if n_out==2
        f_xp = fout{1};
        df_dx = fout{2};
    end
        
    % update decision variables
    mt = beta_1.*mt + (1-beta_1).*df_dx;
    vt = beta_2.*vt + (1-beta_2).*(df_dx.^2);
    mt_p = mt/(1 - beta_1.^t);
    vt_p = vt/(1 - beta_2.^t);
    xp = xp - alpha.* mt_p./(sqrt(vt_p) + 10e-8);
    
    % log information
    if mod(t,log_iter)==0
        fprintf('%5d | %15f | %15f | \n', t, f_xp, alpha)
    end
    
    % check end conditions
    if t>MaxIterations
        break
    end
end
