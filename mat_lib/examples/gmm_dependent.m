function [model, y] = gmm_dependent()
%clear all
clc

n_samples = 1000;

%% generate random data
%sigma_r = 0.4;
sigma_r = 1.0*exp(-0.005*[1:n_samples]);
w_r = [1]; % has to sum to one

train.x = linspace(0, 6, n_samples)';
train.y = zeros(n_samples, 1);

for i=1:n_samples
    train.y(i)= normrnd( sin(train.x(i)), sigma_r(i));
end
plot(train.x,train.y,'x')

%% define model
% parameters
n_inputs = 1;
n_outputs = 1;

n_hidden = [3]; %3 for sigmoid

n_dim = 1;
n_kernels = 1;

n_mu = n_dim * n_kernels;
n_sigma = n_dim * n_kernels;
n_w = n_kernels;

% create model
network= NetworkModel( );

[x]   = network.add_node(ConstantNode());
[z1]  = network.add_node(AffineLayer(n_inputs, n_hidden(1), x));
[h1]  = network.add_node(SigmoidLayer(z1));
[z2]  = network.add_node(AffineLayer(n_hidden(1), n_w + n_mu + n_sigma , h1));
[py_x] = network.add_node(GmmLayerDep(n_dim, n_kernels, z2));
[loss]= network.add_node(NegLogLoss(py_x));

% feed input
x.node.const.x = train.x;
py_x.node.const.y = train.y;

%% compute forward pass
l = network.forward();

%% compute backward pass
dl_dw = network.backward(loss);

%% gradient check
grad_check(@loss_dloss, network.n_params, 200, network, loss);

%% run training

w0 = rand(network.n_params, 1);
f = @(x)loss_dloss(x, network, loss);

options = optimset('Display','iter', 'MaxIter', 100, 'GradObj','on');

[w_p, fval] = fminunc(f, w0, options);

y = loss_eval(w_p, network);


%% graph
if n_dim == 1 && n_kernels == 1
    
    figure(2)
    
    y_plot = zeros(n_samples,1);
    mu_plot = zeros(n_samples,1);
    sigma_plot = zeros(n_samples,1);
        
    for i=1:n_samples
        y_plot(i) = normrnd( py_x.node.mu(i), py_x.node.sigma(i));
        
        mu_plot(i) = py_x.node.mu(i);
        sigma_plot(i) = py_x.node.sigma(i);
        
    end
    
    plot(train.x,train.y,'x')
    hold on
    plot(train.x, y_plot,'ro');
    plot(train.x, mu_plot,'k--', 'LineWidth', 3);
    plot(train.x, mu_plot+sqrt(sigma_plot),'k-', 'LineWidth', 3); % plot of 1 standard deviation
    plot(train.x, mu_plot-sqrt(sigma_plot),'k-', 'LineWidth', 3);
    
    hold off

    figure(1)
end

end



function y = loss_eval(w, model)
    model.vect2params(w);
    y = model.forward();
    
    y = y{end};
end

function y = prob_eval(x, model)
    model.nodes{1}.const.x = x;
    y = model.forward();
    
    y = y{end-1};
end

function dy = dloss_eval(w, model)
    model.vect2params(w);
    y = model.forward();
    dy = backward(model); 
end

function [loss, dloss] = loss_dloss(w, model, target)
    % forward pass
    model.vect2params(w);
    y = model.forward(target);
    loss = y{1};
    
    % backward pass
    %if nargout == 2
    dloss = model.backward(target); 
    %end
end




