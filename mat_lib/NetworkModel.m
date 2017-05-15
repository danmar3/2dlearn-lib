classdef NetworkModel < handle
    % Network class that defines a computation graph
    %
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties
        leaves        
        roots
        
        % TODO(1): define a class for cell inputs, parameters, etc, with
        % function overloading
                        
        % every node saves its parameters (params) and the derivative of
        % its parameters (dparams), each node has to be independent (no
        % copies) 
        % nodes are handles (for memory efficiency), i.e. just references
        % to objects, so be careful when changing their values
        nodes
        n_nodes
        
        %
        params  % cell that saves the "vectorized" parameters for each node
        n_params
        dl_dw   % cell that saves the "vectorized" parameters gradient for each node
        
        evaluated_nodes % list with the nodes that are needed to be evaluated in the forward evaluation
                
        to_backwad_eval % list with the nodes that have to be evaluated
    end
    
    
    methods
        function obj = NetworkModel()
            obj.nodes= cell(0);
            obj.n_nodes= 0;
            
            obj.leaves= cell(0);
            obj.roots= cell(0);
            
            obj.n_params= 0;
        end
        
        
        
        function detect_leaves_roots(obj)
            % clean up
            obj.roots = cell(0);
            obj.leaves = cell(0);
            
            % loop through nodes to detect if they are leaves/roots
            for i= 1:length(obj.nodes)
                node_i = obj.nodes{i};
                % detect roots
                if isempty(node_i.parents)
                    obj.roots{end+1} = node_i;
                end                
                % detect leaves
                if isempty(node_i.childs)
                    obj.leaves{end+1} = node_i;
                end
            end
        end
        
        
        
        function varargout = add_node(obj, new_node) 
            % check if node already was assigned to a network
            assert( new_node.id == -1 , 'duplicate nodes are not allowed' )
            
            % give the node an id and add the node to the list
            new_node.id = obj.n_nodes +1;
            obj.n_nodes = obj.n_nodes + 1;
            obj.nodes{end + 1} = new_node;            
            
            % add parent_node into the parents of node_in and as a child of
            % the parent TODO: delete parents/childs
            for i= 1:length(new_node.inputs)
                parent_node = new_node.inputs{i}.src_port.node;
                % check if parent_node is not already a parent of the node,
                % if it is not a parent, add the node to the parents list
                if ~any(cellfun(@(x) x==parent_node, new_node.parents))
                    assert( parent_node.id ~= -1 , 'parent node has not been added to the network' )
                    assert( parent_node == obj.nodes{parent_node.id} , 'parent node does not coincide with the one saved on the network' )

                    new_node.parents{end+1} = parent_node;
                    parent_node.childs{end+1} = new_node;
                end
            end
            
            % evaluate the number of parameters for the node
            obj.n_params = obj.n_params + new_node.get_num_params();
            
            % detect leaf nodes and roots
            obj.detect_leaves_roots();
            
            % return the references for the node output ports
            varargout = new_node.outputs;
        end
        
        
        
        function stack = get_forward_stack(obj, target_ports)
            % 1. get target nodes from target ports
            assert(any(cellfun(@(x)isa(x, 'NetworkOutputPort'), target_ports)), ...
                   'list of targets must be composed by NetworkOutputPort')
            
            target_nodes{1} = target_ports{1}.node;
            for i=2:length(target_ports)
                node_i = target_ports{i}.node;
                % check that node has not been added to the cell-array
                if ~any( cellfun(@(x)isequal(x, node_i), target_nodes) );  
                    target_nodes = [{node_i}, target_nodes];
                end
            end
            
            % 2. get number of visits for each node in the network
            to_evaluate = target_nodes; % fifo structure
            n_visits= zeros(length(obj.nodes), 1);
            while ~isempty(to_evaluate)
                % dequeue node from to_evaluate:
                node_i= to_evaluate{1};
                to_evaluate= to_evaluate(2:end);
                
                % loop through the parents of node_i
                for m = 1:length(node_i.parents)
                    n_visits( node_i.parents{m}.id ) = n_visits( node_i.parents{m}.id ) + 1;
                    if (n_visits( node_i.parents{m}.id ) == 1)
                        to_evaluate{end+1}= node_i.parents{m};
                    end
                end                
            end
            
            % 3. run topological sort
            stack= cell(0);       % stack must be evaluated in order, i.e. stack{1}->stack{2}->stack{3}
            to_evaluate = target_nodes; % fifo structure
            
            while ~isempty(to_evaluate)
                % dequeue node from to_evaluate:
                node_i= to_evaluate{1};
                to_evaluate= to_evaluate(2:end);
                
                % push node_i into stack
                stack = [{node_i}, stack]; 
                
                % loop through the parents of node_i
                for m = 1:length(node_i.parents)
                    n_visits( node_i.parents{m}.id ) = n_visits( node_i.parents{m}.id ) - 1;
                    if (n_visits( node_i.parents{m}.id ) == 0)
                        to_evaluate{end+1}= node_i.parents{m};
                    end
                end                
            end
            
        end
        
        
        
        function y = forward_stack(obj, stack)
            % TODO: reset 
            obj.evaluated_nodes= false(length(obj.nodes), 1);
            
            for i= 1:length(stack)
                node_i = stack{i};
                y = cell(node_i.n_outputs, 1);
                
                if isempty(node_i.parents) % if root
                    [y{:}] = node_i.forward();
                                        
                else % if depends on other nodes 
                    % 1. obtain inputs from the ports
                    input_list = cell(node_i.n_inputs, 1);
                    for j= 1:node_i.n_inputs
                        inport_j = node_i.inputs{j};
                        % TODO(1): critical section, extensive test needed
                        % for multiple inputs/outputs
                        assert(obj.evaluated_nodes( inport_j.src_port.node.id ), 'parent has not been evaluated')
                        input_list{j}= inport_j.src_port.data;
                    end
                    
                    % 2. execute node
                    [y{:}] = node_i.forward(input_list{:});
                    
                end
                
                % update related ports 
                for out_i= 1:node_i.n_outputs
                    % update outputs of the node
                    node_i.outputs{out_i}.set_data(y{out_i});
                    
                    % update inputs of the child nodes
                    for dest_i= 1:length(node_i.outputs{out_i}.dest_ports)
                        dest_port = node_i.outputs{out_i}.dest_ports{dest_i};
                        dest_port.set_data(y{out_i});
                    end
                end

                obj.evaluated_nodes(node_i.id) = true;
                
            end
            
        end
        
        
        function [y, call_stack] = forward( obj, varargin ) 
            if nargin==1
                valid_nodes= obj.leaves(find(cellfun(@(x)not(isa(x, 'VariableNode')|isa(x, 'ConstantNode')), ...
                                        obj.leaves)));
                %target_ports = valid_nodes(1).outputs(1);
                target_ports = cellfun(@(x)(x.outputs{1}), valid_nodes, 'UniformOutput', false);
                disp('target nodes:')
                disp(valid_nodes)
            else
                target_ports = varargin;
            end
            call_stack= obj.get_forward_stack(target_ports);
            
            obj.forward_stack(call_stack);
            y = cellfun(@(x)(x.data), target_ports, 'UniformOutput', false);
        end
        
        
        
        function w = params2vect(obj)            
            if isempty(obj.params)
                obj.params = cell(obj.n_nodes, 1);
                %for i= 1:obj.n_nodes
                %    node_i = obj.nodes{i};
                %    obj.params{i} = zeros( node_i.n_params, 1);
                %end
            end
            
            % copy parameters into params cell
            for i= 1:obj.n_nodes
                node_i = obj.nodes{i};
                obj.params{i} = node_i.params2vect();
            end
            
            % TODO: look for a more efficient memory usage for this
            w = cell2mat(obj.params);
        end
        
        
        
        function vect2params(obj, w)
            if isempty(obj.params)
                obj.params2vect();
            end
            param_size = cell2mat(cellfun(@(x) numel(x), obj.params, 'UniformOutput', false));
            param = mat2cell(w, param_size, 1);
            
            for i=1:obj.n_nodes
                obj.nodes{i}.vect2params(param{i});
            end
            
        end
        
        
        
        function stack = get_backward_stack(obj, node)
            
            stack_forward = obj.get_forward_stack({node});
            % reverse stack_forward
            stack_aux = cell(length(stack_forward), 1);
            for i= length(stack_forward):-1:1
                stack_aux{i} = stack_forward{ length(stack_forward) - i + 1 } ;
            end
            
            % remove nodes that do not propagate the gradient
            eval_nodes = zeros(obj.n_nodes, 1);
            eval_nodes(stack_aux{1}.id) = stack_aux{1}.propagate_gradients;
            if stack_aux{1}.propagate_gradients
                stack = stack_aux(1);
            else
                error('Target node does not propagate gradients \n')
            end
            
            for i = 2:length(stack_aux)
                eval_node_i = any(cellfun( @(x) eval_nodes(x.id), stack_aux{i}.childs));
                eval_node_i = and(eval_node_i, stack_aux{i}.propagate_gradients);
                if eval_node_i
                    eval_nodes(stack_aux{i}.id) = true;
                    stack{end+1} = stack_aux{i};
                end
            end
            
        end
        
        
        
        function dl_dw = backward_stack(obj, stack, alpha)
            % TODO(1): improve for multiple inputs/outputs
            if nargin==2
                alpha = 1;
            end
            
            % 1. initialize derivatives to zero
            if isempty(obj.dl_dw)
                obj.dl_dw = cell(obj.n_nodes, 1);
            end
            for i= 1:obj.n_nodes
                node_i = obj.nodes{i};
                node_i.reset_gradient();
                
                obj.dl_dw{i} = zeros(node_i.n_params, 1);
            end
            
            
            % 2. evaluate initial node
            node_i = stack{1};
            if node_i.n_params ~= 0
                obj.dl_dw{node_i.id} = node_i.serialize_params(node_i.backward_params(alpha));
            end
            % calculate gradient on inputs
            dl_dx = cell(node_i.n_inputs, 1);
            [dl_dx{:}] = node_i.backward_inputs(alpha);
            for k = 1:node_i.n_inputs
                % update input port
                node_i.inputs{k}.dl_dx = dl_dx{k};
                % update source output-port
                node_i.inputs{k}.src_port.dl_dy = node_i.inputs{k}.src_port.dl_dy + dl_dx{k};
            end
            
                        
            % 3. evaluate all nodes in stack
            for i= 2:length(stack)
                node_i = stack{i};
                
                %3.1 gather dl_dy from the output ports
                dl_dy = cell(node_i.n_outputs, 1);
                for k = 1:node_i.n_outputs
                    dl_dy{k} = node_i.outputs{k}.dl_dy;
                end
                                
                % 3.2. calculate parameters gradient
                if node_i.n_params ~= 0
                    obj.dl_dw{node_i.id} = node_i.serialize_params( node_i.backward_params(dl_dy{:}) );
                end
                
                % 3.3. calculate inputs gradient
                if node_i.n_inputs ~= 0 % if not a root
                    dl_dx = cell(node_i.n_inputs, 1);
                    [dl_dx{:}] = node_i.backward_inputs(dl_dy{:});
                    for k = 1:node_i.n_inputs
                        % update input port
                        node_i.inputs{k}.dl_dx = dl_dx{k};
                        % update source output-port
                        node_i.inputs{k}.src_port.dl_dy = node_i.inputs{k}.src_port.dl_dy + dl_dx{k};
                    end
                end
                                        
            end
            
            dl_dw = cell2mat(obj.dl_dw);
            
        end
        
        
        
        function [dl_dw, stack] = backward(obj, target_port, alpha)
            % target_node: ussualy a loss, this is the node from wich the
            %              derivative is evaluated
            % alpha : a constant which is fed into the target_node backprop
            %
            % the backward computation must start from a node that outputs
            % an scalar
            assert(isa(target_port, 'NetworkOutputPort'), ...
                   'Specified target is not a NetworkOutputPort')
               
            if nargin<2
                valid_nodes= find(cellfun(@(x)not(isa(x, 'VariableNode')|isa(x, 'ConstantNode')), ...
                                  obj.leaves));
                target_port = obj.leaves{valid_nodes(1)}.outputs{1};
                disp('target port:')
                disp(target_port)
                stack = obj.get_backward_stack(target_port);
            else
                stack = obj.get_backward_stack(target_port);
            end
            
            if nargin == 3
                dl_dw = obj.backward_stack(stack, alpha);
            else
                dl_dw = obj.backward_stack(stack, 1);
            end
            
        end
        
        function list_names(obj)
            fprintf('Network output ports names: \n')
            for i=1:length(obj.nodes)
                fprintf('%i: %s', i, obj.nodes{i}.outputs{1}.name)
                for j=2:length(obj.nodes{i}.outputs)
                    fprintf(', %s:', obj.nodes{i}.outputs{j}.name)
                end
                fprintf('\n')
            end
        end
        
        function data = get_tensor(obj, name)
            %TODO: consider using hashing for this
            for i = 1:length(obj.nodes)
                for j = 1:length(obj.nodes{i}.outputs)
                    if strcmp( obj.nodes{i}.outputs{j}.name, name )
                        data = obj.nodes{i}.outputs{j}.data;
                        return
                    end
                end
            end
            data = 0;
            fprintf('Not such variable with given name \n');
        end
        
    end
    
end