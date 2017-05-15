function [network, y] = gmm_network()
clear 
clc

n_samples = 100;

%% generate random data
sigma_r = [0.05];
w_r = [1]; % has to sum to one

train.x = linspace(0, 6, n_samples)';
train.y = zeros(n_samples, 1);

for i=1:n_samples
    train.y(i)= normrnd( sin(train.x(i)), sigma_r);

end
plot(train.x,train.y,'x')

%w_r = permute(w_r, [1 3 2] );

%% define model
% parameters
n_inputs = 1;
n_outputs = 1;

n_hidden = [5];

% layers
layers{1}= AffineLayer(n_inputs, n_hidden(1));
layers{2}= SigmoidLayer();

layers{3}= AffineLayer(n_hidden(1), 1);

layers{4}= L2Loss();

% create model
network= NetworkModel( );

network.add_node(layers{1});
network.add_node(layers{2}, layers{1});
network.add_node(layers{3}, layers{2});
network.add_node(layers{4}, layers{3});


%model.layers{1}.params.mu = rand(1, n_dim, n_kernels);
%model.layers{1}.params.sigma = rand(1, n_dim, n_kernels);
%model.layers{1}.params.w = 1; %rand(1, 1, n_kernels);

layers{1}.const.x = train.x;
layers{4}.const.y = train.y;

w = network.params2vect();


%% compute forward pass
y = network.forward();

%% compute backward pass
dy_dw = network.backward(layers{4});


%% gradient check
grad_check(@loss_dloss, network.n_params, 200, network);

%% run training

w0 = rand(network.n_params, 1);
f = @(x)loss_dloss(x, network);

options = optimset('Display','iter', 'MaxIter', 100, 'GradObj','on');

[w_p, fval] = fminunc(f, w0, options);

y = loss_eval(w_p, network);


%% graph
if n_inputs == 1
    figure(2)
    
    y_plot= prob_eval(train.x, network);
    
    plot(train.x,train.y,'x')
    hold on
    plot(train.x, y_plot,'g-');
    hold off
        
    figure(1)
end


end



function y = loss_eval(w, model)
    model.vect2params(w);
    y = model.forward();
    
    %y = y{end};
end

function y = prob_eval(x, network)
    network.nodes{1}.const.x = x;
    y = network.forward(network.nodes{3});
    
    %y = y{end-1};
end

function dy = dloss_eval(w, model)
    model.vect2params(w);
    y = model.forward();
    dy = backward(model); 
end

function [loss, dloss] = loss_dloss(w, model)
    % forward pass
    model.vect2params(w);
    loss = model.forward();
        
    % backward pass
    %if nargout == 2
    dloss = model.backward(); 
    %end
end