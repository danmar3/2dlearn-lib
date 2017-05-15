classdef ExpNode < NetworkNode
    % ExpNode: node that computes the exponential function exp(x)
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        y
    end
    methods 
        
        function obj = ExpNode(x)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            obj.y = exp(x);
            y = obj.y;
        end
               
        function dl_dx = backward_inputs(obj, de_dy)
            dl_dx = de_dy .* obj.y;
        end
        
        
    end
    
end