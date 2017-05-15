function [network, y] = mlp_basic()
clear all
clc

n_samples = 1000;

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

n_hidden = [100, 10];

% create model
global network
network= NetworkModel( );

init_mul = 0.1;
x   = ConstantNode().atn();
yd  = ConstantNode().atn();
w1  = VariableNode([n_hidden(1), n_inputs], init_mul).atn();
b1  = VariableNode([1, n_hidden(1)], init_mul).atn();
w2  = VariableNode([n_outputs, n_hidden(1)], init_mul).atn();
b2  = VariableNode([1, n_outputs], init_mul).atn();

%w3  = VariableNode([n_hidden(2), n_outputs], init_mul).atn();
%b3  = VariableNode([1, n_outputs], init_mul).atn();

h1  = 0.5.*(relu(x*w1' + b1)).^2;
z2  = h1*w2' + b2;
loss= reduce_sum((z2-yd).^2, [1]);

% feed inputs
x.feed((train.x - mean(train.x))/std(train.x));
yd.feed(train.y);

w = network.params2vect();

%% compute forward pass
y = network.forward(loss);

%% compute backward pass
dy_dw = network.backward(loss);


%% gradient check
grad_check(@loss_dloss, network.n_params, 200, network, loss);

%% run training

w0 = rand(network.n_params, 1);
f = @(x)loss_dloss(x, network, loss);

%options = optimset('Display','iter', 'MaxIter', 50, 'GradObj','on');
options = optimoptions(@fminunc,'Algorithm','quasi-newton', ...
                                'Display','iter', ...
                                'MaxIterations', 1000, ...
                                'SpecifyObjectiveGradient', true);
                            
[w_p, fval] = fminunc(f, w0, options);

y = loss_eval(w_p, network, loss);


%% graph
if n_inputs == 1
    figure(2)
    x_feed= (train.x - mean(train.x))/std(train.x);
    y_plot= prob_eval(x_feed, z2);
    
    plot(train.x,train.y,'x')
    hold on
    plot(train.x, y_plot,'g-');
    hold off
        
    figure(1)
end


end



function y = loss_eval(w, model, target)
    model.vect2params(w);
    y = model.forward(target);
    
    y = y{1};
end

function y = prob_eval(x, target)
    global network
    network.nodes{1}.const.x = x;
    y = network.forward(target);
    
    y = y{1};
end

function dy = dloss_eval(w, model)
    model.vect2params(w);
    model.forward();
    dy = backward(model); 
end

function [loss, dloss] = loss_dloss(w, model, target)
    % forward pass
    model.vect2params(w);
    out = model.forward(target);
    loss= out{1};    
    % backward pass
    %if nargout == 2
    dloss = model.backward(target); 
    %end
end