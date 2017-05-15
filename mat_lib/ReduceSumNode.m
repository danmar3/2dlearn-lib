classdef ReduceSumNode < NetworkNode
    % ReduceSumNode: node that computes the sum of elements across 
    %                dimensions of a tensor
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
properties ( Access = private )
        reduce_dims
        size_x
    end
    
    methods 
        function obj = ReduceSumNode(x, reduce_dims)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            obj.reduce_dims = reduce_dims;
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            obj.size_x = size(x);
            
            y = sum( x, obj.reduce_dims(1));
            for i= 2:length( obj.reduce_dims )
                y = sum( y, obj.reduce_dims(i));
            end
        end
        
        function dl_dx = backward_inputs(obj, de_dy)
            dl_dx = bsxfun( @times, de_dy , ones(obj.size_x));
        end
    end    
end