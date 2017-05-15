function [network, y] = gmm_constrained()
%clear all
clc

n_samples = 100;

%% generate random data
mu_r = [-6, 8];
sigma_r = [1.6, 1.3; 1.3, 1.3];
w_r = [1]; % has to sum to one

train.x = mvnrnd(mu_r, sigma_r, n_samples);
%{
mu_r = [-10, 10];
sigma_r = [1, 0 ; 0 ,1];
train.x = [train.x ; mvnrnd(mu_r, sigma_r, n_samples)];
%}

mu_r = [-6, 8];
sigma_r = [1.6, -1.3; -1.3, 1.3];
train.x = [train.x ; mvnrnd(mu_r, sigma_r, n_samples)];


mu_r = [-10, 10];
sigma_r = [3, 0 ; 0 ,5];
train.x = [train.x ; mvnrnd(mu_r, sigma_r, n_samples)];


% normalize
train.x = bsxfun( @times, (bsxfun(@plus, train.x, -mean(train.x))), 1./std(train.x));

% plot
figure(1)
plot(train.x(:,1),train.x(:,2),'x')

w_r = permute(w_r, [1 3 2] );

% conf parameters
n_dim = 2;
n_kernels = 3;

%% define model
network= NetworkModel( );

[p_x, q] = network.add_node(GmmLayer(n_dim, n_kernels, 'constrained'));
[loss]= network.add_node(NegLogLoss(p_x));
[loss_c]= network.add_node(OrthogonalLoss(q));


%model.layers{1}.params.w = 1; %rand(1, 1, n_kernels);

p_x.node.const.x = train.x;

%% compute forward pass
y = network.forward();

%% compute backward pass
dy_dw = network.backward(loss);

%% gradient check
n_params= network.n_params;

grad_check(@loss_dloss, n_params, 200, network, loss);
grad_check(@const_dconst, n_params, 200, network, loss_c);

%% run training

w0 = network.params2vect();%rand(n_params, 1);
f = @(x)loss_dloss(x, network, loss);
c = @(x)const_dconst(x, network, loss_c);

%{
% for matlab 2012
%options = optimset('Display','iter', 'MaxIter', 100, 'GradObj','on');
%[x, fval] = fminunc(f, w0, options);

% for matlab 2016
options = optimoptions(@fminunc,'Display', 'iter', ...
                       'MaxIterations', 100, ...
                       'SpecifyObjectiveGradient',true, ...
                       'Algorithm','quasi-newton'); % quasi-newton, trust-region
[x, fval] = fminunc(f, w0, options);
%}

% for matlab 2012
options = optimset('Display', 'iter', ...
                   'MaxIter', 300, ...
                   'GradObj', 'on', ...
                   'GradConstr', 'on', ...
                   'Algorithm','interior-point'); % interior-point, trust-region-reflective
[x, fval] = fmincon(f, w0,[],[],[],[],[],[],c,options);
%{
% for matlab 2016
options = optimoptions('fmincon','Display', 'iter', ...
                       'MaxIterations', 300, ...
                       'SpecifyObjectiveGradient',true, ...
                       'SpecifyConstraintGradient',true, ...
                       'Algorithm','interior-point'); % interior-point, trust-region-reflective
[x, fval] = fmincon(f, w0,[],[],[],[],[],[],c,options);
%}
y = loss_eval(x, network, loss);

%% graph
if n_dim == 2
    figure(2)
    dx= 0.1;
    [x1aux,x2aux] = meshgrid([-2:dx:2],[-2:dx:2]);
    
    caux=cat(2,x1aux',x2aux');
    X_plot=reshape(caux,[],2);

    y_plot= prob_eval(X_plot, network, p_x);
    y_plot= reshape(y_plot, size(x1aux,2), size(x1aux,1))';

    mesh(x1aux,x2aux,y_plot)
    hold on
    plot(train.x(:,1), train.x(:,2),'x');
    contour(x1aux,x2aux,y_plot,[0 0])
    hold off
    
    figure(3)
    contour(x1aux,x2aux,y_plot, 20)
    hold on
    plot(train.x(:,1), train.x(:,2),'x');
    hold off
    figure(1)
end

end

function y = loss_eval(w, network, target)
    network.vect2params(w);
    y = network.forward(target);
    
end

function y = prob_eval(x, network, target)
    network.nodes{1}.const.x = x;
    y = network.forward(target);
    y = y{1};
end

function [loss, dloss] = loss_dloss(w, network, target)
    % forward pass
    network.vect2params(w);
    y = network.forward(target);
    loss = y{1};
    % backward pass
    %if nargout == 2
    dloss = network.backward(target); 
    %end
end

function [c, ceq, dc, dceq] = const_dconst(w, network, target)
    % forward pass
    network.vect2params(w);
    y = network.forward(target);
    ceq = y{1};
    % backward pass
    dceq = network.backward(target); 
    %
    c = [];
    dc= [];
end