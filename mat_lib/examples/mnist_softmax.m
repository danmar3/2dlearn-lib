function network = mnist_softmax()
clc

n_samples = 5000;
max_steps = 3000;

%% 1. Load dataset
mnist = tdl.datasets.MnistDataset();

%% 2. Define the network
% parameters
n_inputs = 28*28;
n_hidden = [10];

% create model
global network
network= NetworkModel( );

init_mul = 1;
x   = ConstantNode().atn();

w1  = VariableNode([n_inputs, n_hidden(1)], init_mul).atn();        w1.name = 'w1';
b1  = VariableNode([1, n_hidden(1)], init_mul).atn();               b1.name = 'b1';
% c1  = VariableNode([1, n_inputs], init_mul).atn();                  c1.name = 'c1';
% w2  = VariableNode([n_hidden(1), n_hidden(2)], init_mul).atn();     w2.name = 'w2';
% b2  = VariableNode([1, n_hidden(2)], init_mul).atn();               b2.name = 'b2';
% c2  = VariableNode([1, n_hidden(1)], init_mul).atn();               c2.name = 'c2';
% w3  = VariableNode([n_hidden(2), n_hidden(3)], init_mul).atn();     w3.name = 'w3';
% b3  = VariableNode([1, n_hidden(3)], init_mul).atn();               b3.name = 'b3';
% c3  = VariableNode([1, n_hidden(2)], init_mul).atn();               c3.name = 'c3';
% w4  = VariableNode([n_hidden(3), n_hidden(4)], init_mul).atn();     w4.name = 'w4';
% b4  = VariableNode([1, n_hidden(4)], init_mul).atn();               b4.name = 'b4';
% c4  = VariableNode([1, n_hidden(3)], init_mul).atn();               c4.name = 'c4';

%w3  = VariableNode([n_hidden(2), n_outputs], init_mul).atn();
%b3  = VariableNode([1, n_outputs], init_mul).atn();
%h{1}  = sigmoid(x*w1 + b1);                         h{1}.name = 'h1';
%h{2}  = h{1}*w2 + b2;                               h{2}.name = 'h2';

h1 = x*w1 + b1;
y  = SoftmaxNode(h1).atn();
loss_c = SoftmaxCrossentropyWithLogitsNode(h1).atn();
loss = loss_c ; %+ 10.*(reduce_sum(w1.^2, [1, 2]));

% feed inputs
[train_x, train_y] = mnist.train.next_batch(n_samples);
x.feed(train_x);
loss_c.node.const.y = train_y;


%% gradient check
%loss_dloss_g = @(w) tdl.loss_dloss(w, network, loss);
%grad_check(loss_dloss_g, network.n_params, 200);


%% training function
iter = 0;
    function [f, df] = loss_dloss(w, model, target)        
        % {
        % 1. get next batch
        [train_x, train_y] = mnist.train.next_batch(n_samples);
        x.feed(train_x);
        loss_c.node.const.y = train_y;
        % }
        % 2. forward pass
        model.vect2params(w);
        out = model.forward(target);
        f= out{1};
        % 3. backward pass
        df = model.backward(target);
        % {
        % 4. logging
        if mod(iter, 100)==0
            % training
            out = model.forward(y);
            [~, p_class] = max(out{1}, [], 2);
            [~, d_class] = max(train_y, [], 2);
            fprintf('accuracy (training, %d): %f \n', mnist.train.epochs_completed, sum(p_class == d_class)/length(d_class));
            
            % testing
            [test_x, test_y] = mnist.test.next_batch(n_samples);
            x.feed(test_x);

            % forward pass
            out = model.forward(y);
            [~, p_class] = max(out{1}, [], 2);
            [~, d_class] = max(test_y, [], 2);
            fprintf('accuracy (testing): %f \n', sum(p_class == d_class)/length(d_class));
        end
        % }
        iter = iter + 1;
    end


%% run training

w0 = network.params2vect();
%f = @(x)tdl.loss_dloss(x, network, loss_r1);

f = @(x)loss_dloss(x, network, loss);
        
%{
options = optimoptions(@fminunc,'Algorithm','quasi-newton', ...
                                'Display','iter', ...
                                'MaxIterations', max_steps, ...
                                'SpecifyObjectiveGradient', true);

[w_p, fval] = fminunc(f, w0, options);
%}
% {
options = tdl.optim.optimoptions(@tdl.optim.sgd,...
                 'Display', 100, ...
                 'MaxIterations', max_steps, ...
                 'LearningRate', @(i) 0.1*0.9^(i/ 500), ... % learning_rate*decay_rate^(global_step / decay_steps) , 100
                 'Momentum', 0.5, ...
                 'MaxValue', 10e10 ...
                );
[w_p, fval] = tdl.optim.momentum_optimizer(f, w0, options);
w0 = w_p;
% }
%[y, ~] = f(w_p);


%% Get accuracy on testing dataset
[test_x, test_y] = mnist.test.next_batch(n_samples);
x.feed(test_x);
        
% forward pass
network.vect2params(w_p);
out = network.forward(y);
[~, p_class] = max(out{1}, [], 2);
[~, d_class] = max(test_y, [], 2);
fprintf('accuracy (testing): %f \n', sum(p_class == d_class)/length(d_class));


end