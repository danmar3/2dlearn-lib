classdef ReduceMeanNode < NetworkNode
    % ReduceMeanNode: node that computes the mean of elements across 
    %                dimensions of a tensor
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
properties ( Access = private )
        reduce_dims
        size_x
        m
    end
    
    methods 
        function obj = ReduceMeanNode(x, reduce_dims)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            obj.reduce_dims = reduce_dims;
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            obj.size_x = size(x);
            obj.m = 1;
            y = x;
            for i= obj.reduce_dims
                y = sum( y, i);
                obj.m = obj.m * obj.size_x(i);
            end
            y = (1/obj.m)*y;
        end
        
        function dl_dx = backward_inputs(obj, de_dy)
            dl_dx = bsxfun( @times, de_dy , (1/obj.m)*ones(obj.size_x));
        end
    end    
end