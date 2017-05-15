classdef SoftplusNode < NetworkNode
    % SoftplusNode: computes the softplus function ln(1 + exp(x))
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        x
    end
    methods 
        
        function obj = SoftplusNode(x)           
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            obj.x = x;
            y = log(1 + exp(x));
        end
                
        function dl_dx = backward_inputs(obj, de_dy)
            dl_dx = de_dy .* (1./(1+exp(-obj.x)));
        end
        
        
    end
    
end