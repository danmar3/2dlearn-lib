classdef SoftmaxNode < NetworkNode
    % SoftmaxNode: layer for a feedforward network that computes the
    %              softmax function
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        y
    end
    methods 
        
        function obj = SoftmaxNode(x)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            obj.y = bsxfun(@rdivide, exp(x), sum(exp(x), 2) );
            y = obj.y;
        end
        
        function backward_params(obj, de_dy)
            
        end
        
        function dl_dx = backward_inputs(obj, de_dy)
            error('Not implemented yet');
        end
        
        
    end
    
end