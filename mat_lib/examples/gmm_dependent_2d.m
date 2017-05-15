function [model, y] = gmm_dependent_2d()
%clear all
clc

n_samples = 1000;

%% generate random data
x_dim = 2;
es = 0.001;
em = 0.01;
train.x = 0.5*ones(n_samples, x_dim);

train.y = es*mvnrnd(zeros(x_dim, 1), eye(x_dim), n_samples) + ...
          bsxfun(@times, mvnrnd(ones(x_dim, 1), em*eye(x_dim), n_samples), train.x);

y_norm = sqrt(sum(train.y.^2, 2));
train.y = bsxfun(@times, train.y, (y_norm.^2)./y_norm);


%y_norm = sqrt(sum(train.y.^2, 2));
%train.y = bsxfun(@times, train.y, 1.0*tanh(y_norm)./y_norm);

plot(train.y(:,1), train.y(:,2), 'o')

hold on

%train.x = 1*ones(n_samples, x_dim);
train.x = mvnrnd(ones(x_dim, 1), 3*eye(x_dim), n_samples);
train.y = es*mvnrnd(zeros(x_dim, 1), eye(x_dim), n_samples) + ...
          bsxfun(@times, mvnrnd(ones(x_dim, 1), em*eye(x_dim), n_samples), train.x);


y_norm = sqrt(sum(train.y.^2, 2));
train.y = bsxfun(@times, train.y, (y_norm.^2)./y_norm);


plot(train.y(:,1), train.y(:,2), 'ro')



hold off


%pause
%{
%sigma_r = 0.4;
sigma_r = 1.0*exp(-0.005*[1:n_samples]);
w_r = [1]; % has to sum to one

train.x = linspace(0, 6, n_samples)';
train.y = zeros(n_samples, 1);

for i=1:n_samples
    train.y(i)= normrnd( sin(train.x(i)), sigma_r(i));
end
plot(train.x,train.y,'x')
%}

%% define model
% parameters
n_inputs = 2;
n_outputs = 2;

n_hidden = [30]; %10 for sigmoid

n_dim = n_outputs; % TODO: redundant with n_outputs
n_kernels = 1;

n_mu = n_dim * n_kernels;
n_sigma = n_dim * n_kernels;
n_w = n_kernels;

% create model
network= NetworkModel( );

[x]   = network.add_node(ConstantNode());
[z1]  = network.add_node(AffineLayer(n_inputs, n_hidden(1), x));
[h1]  = network.add_node(SigmoidLayer(z1));
%[h1]  = network.add_node(ReluLayer(z1));
[z2]  = network.add_node(AffineLayer(n_hidden(1), n_w + n_mu + n_sigma , h1));
[py_x]= network.add_node(GmmLayerDep(n_dim, n_kernels, z2));
[loss]= network.add_node(NegLogLoss(py_x));

% feed input
x.node.const.x = train.x;
py_x.node.const.y = train.y;

%% compute forward pass
l = network.forward(loss);

%% compute backward pass
dl_dw = network.backward(loss);

%% gradient check
grad_check(@loss_dloss, network.n_params, 200, network, loss);

%% run training

w0 = rand(network.n_params, 1);
f = @(x)loss_dloss(x, network, loss);

%options = optimset('Display','iter', 'MaxIter', 200, 'GradObj','on');
options = optimoptions(@fminunc,'Algorithm','quasi-newton', ...
                                'Display','iter', ...
                                'MaxIterations', 1000, ...
                                'SpecifyObjectiveGradient', true);
                            
[w_p, fval] = fminunc(f, w0, options);

y = loss_eval(w_p, network, loss);


%% graph
figure(2)

n_samples = 100;

test.x = 0.5*ones(n_samples, x_dim);
test = run_sym(test);
test_eval(test, network)

hold on
test.x = 1.0*ones(n_samples, x_dim);
test = run_sym(test);
test_eval(test, network)
hold off


figure(1)

end



function y = loss_eval(w, model, target)
    model.vect2params(w);
    y = model.forward(target);
    
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

function data = run_sym(data)
    % Runs deterministic function to obtain y, given x
    x_dim = 2;
    n_samples = size(data.x, 1); 
    es = 0.001;
    em = 0.01;

    data.y = es*mvnrnd(zeros(x_dim, 1), eye(x_dim), n_samples) + ...
              bsxfun(@times, mvnrnd(ones(x_dim, 1), em*eye(x_dim), n_samples), data.x);

    y_norm = sqrt(sum(data.y.^2, 2));
    data.y = bsxfun(@times, data.y, (y_norm.^2)./y_norm);
    
end

function y = test_eval(test, network)
    plot(test.y(:,1), test.y(:,2), 'o')
    
    x = network.nodes{1}.outputs{1};
    py_x = network.nodes{5}.outputs{1};
    
    x.node.const.x = test.x;
    py_x.node.const.y = test.y;
    y = network.forward();

	hold on

    mu = squeeze(py_x.node.mu(1,1,:))'; %// data
    sigma = diag(squeeze(py_x.node.sigma(1,1,:))); %// data
    plot_x = -2.0:.01:2.0; %// x axis
    plot_y = -2.0:.01:2.0; %// y axis

    [X, Y] = meshgrid(plot_x, plot_y); %// all combinations of x, y
    Z = mvnpdf([X(:) Y(:)],mu,sigma); %// compute Gaussian pdf
    Z = reshape(Z,size(X)); %// put into same size as X, Y
    contour(X,Y,Z), axis equal  %// contour plot; set same scale for x and y...
    %surf(X,Y,Z) %// ... or 3D plot
    
    hold off
    
end