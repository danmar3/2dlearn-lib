function [network, y] = mlp_network()
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

%% define model
% parameters
n_inputs = 1;
n_outputs = 1;

n_hidden = [5];

% create model
network= NetworkModel( );

[x]   = network.add_node(ConstantNode());
[z1]  = network.add_node(AffineLayer(n_inputs, n_hidden(1), x));
[h1]  = network.add_node(ReluLayer(z1));
[z2]  = network.add_node(AffineLayer(n_hidden(1), 1, h1));
[loss]= network.add_node(L2Loss(z2));

% feed inputs
x.node.const.x = (train.x - mean(train.x))/std(train.x);
loss.node.const.y = train.y;

w = network.params2vect();


%% compute forward pass
y = network.forward();

%% compute backward pass
dy_dw = network.backward(loss.node);


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
    x_feed= (train.x - mean(train.x))/std(train.x);
    y_plot= prob_eval(x_feed, network);
    
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
    
    y = y{1};
end

function y = prob_eval(x, network)
    network.nodes{1}.const.x = x;
    y = network.forward(network.nodes{4});
    
    y = y{1};
end

function dy = dloss_eval(w, model)
    model.vect2params(w);
    model.forward();
    dy = backward(model); 
end

function [loss, dloss] = loss_dloss(w, model)
    % forward pass
    model.vect2params(w);
    out = model.forward();
    loss= out{1};    
    % backward pass
    %if nargout == 2
    dloss = model.backward(); 
    %end
end