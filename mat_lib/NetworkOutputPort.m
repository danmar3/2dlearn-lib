classdef NetworkOutputPort < handle
    % Output port class for a node in the computation graph
    %
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties
        node % 
        name % name of the port's variable
        data
        dl_dy % ordered derivative of the output variable
        dest_ports
        waiting % true if the port is waiting for getting the data
    end
    
    
    methods
        function obj = NetworkOutputPort(node, name)
            obj.node = node;
            obj.name = name;
            obj.waiting = true;
            obj.dest_ports = cell(0);
        end
        
        function x = get_data( obj )
            x = obj.data;
        end
        
        function set_data( obj, x )
            obj.data= x;
            obj.waiting = false;
        end
        
        function add_dest(obj, dest_port)
            obj.dest_ports{end + 1} = dest_port;
        end
        
        function feed(obj, data)
            obj.node.feed(data);
        end        
        
        % ------------------ Operator overloading ---------------- %
        function y = plus(obj1,obj2)
            global network
            if ~isa(obj1, 'NetworkOutputPort')
                obj1 = mat2constnode(obj1);
            end
            if ~isa(obj2, 'NetworkOutputPort')
                obj2 = mat2constnode(obj2);
            end
            y = network.add_node(PlusNode(obj1, obj2));
        end
        
        function y = minus(obj1, obj2)
            global network
            if ~isa(obj1, 'NetworkOutputPort')
                obj1 = mat2constnode(obj1);
            end
            if ~isa(obj2, 'NetworkOutputPort')
                obj2 = mat2constnode(obj2);
            end
            y = network.add_node(SubtractNode(obj1, obj2));
        end
        
        function y = times(obj1, obj2)
            global network
            if ~isa(obj1, 'NetworkOutputPort')
                obj1 = mat2constnode(obj1);
            end
            if ~isa(obj2, 'NetworkOutputPort')
                obj2 = mat2constnode(obj2);
            end
            y = network.add_node(TimesNode(obj1, obj2));
        end
        
        function y = mtimes(obj1, obj2)
            global network
            if ~isa(obj1, 'NetworkOutputPort')
                obj1 = mat2constnode(obj1);
            end
            if ~isa(obj2, 'NetworkOutputPort')
                obj2 = mat2constnode(obj2);
            end
            y = network.add_node(MatmulNode(obj1, obj2));
        end
        
        function y = power(obj1, obj2)
            global network
            y = network.add_node(PowNode(obj1, obj2));
        end
        
        function y = ctranspose(obj)
            global network
            y = network.add_node(TransposeNode(obj));
        end
        
        function y = relu(obj)
            global network
            y = network.add_node(ReluLayer(obj));
        end
        
        function y = sigmoid(obj)
            global network
            y = network.add_node(SigmoidLayer(obj));
        end
        
        function y = softplus(obj)
            global network
            y = network.add_node(SoftplusNode(obj));
        end
        
        function y = exp(obj)
            global network
            y = network.add_node(ExpNode(obj));
        end
        
        function y = log(obj)
            global network
            y = network.add_node(LogNode(obj));
        end
        
        function y = reduce_sum(obj, reduce_dims)
            global network
            y = network.add_node(ReduceSumNode(obj, reduce_dims));
        end
        
        function y = reduce_mean(obj, reduce_dims)
            global network
            y = network.add_node(ReduceMeanNode(obj, reduce_dims));
        end
        
    end
    
end