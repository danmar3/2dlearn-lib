function [network, y] = gmm_unconstrained()
%clear all
clc

n_samples = 100;

%% generate random data
mu_r = [1, 3];
sigma_r = [1.5, 1.3; 1.3, 1.5];
w_r = [1]; % has to sum to one

train.x = mvnrnd(mu_r, sigma_r, n_samples);

mu_r = [-10, 10];
sigma_r = [1, 0 ; 0 ,1];
train.x = [train.x ; mvnrnd(mu_r, sigma_r, n_samples)];


mu_r = [-20, 10];
sigma_r = [3, 0 ; 0 ,7];
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

[p_x] = network.add_node(GmmLayer(n_dim, n_kernels, 'full'));
[loss]= network.add_node(NegLogLoss(p_x));


%model.layers{1}.params.w = 1; %rand(1, 1, n_kernels);

p_x.node.const.x = train.x;

%% compute forward pass
y = network.forward();

%% compute backward pass
dy_dw = network.backward(loss);

%% gradient check
n_params= network.n_params;

grad_check(@loss_dloss, n_params, 200, network, loss);

%% run training

w0 = rand(n_params, 1);
f = @(x)loss_dloss(x, network, loss);

options = optimset('Display','iter', 'MaxIter', 100, 'GradObj','on');

[x, fval] = fminunc(f, w0, options);

y = loss_eval(x, network);

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

    figure(1)
end

end



function y = loss_eval(w, network)
    network.vect2params(w);
    y = network.forward();
    
end

function y = prob_eval(x, network, target)
    network.nodes{1}.const.x = x;
    y = network.forward(target);
    y = y{1};
end

function dy = dloss_eval(w, network)
    network.vect2params(w);
    y = network.forward();
    dy = backward(network); 
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