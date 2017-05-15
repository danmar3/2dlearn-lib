classdef PowNode < NetworkNode
    % PowNode: node that computes the power of one value to another x.^n
    % 
    % Wrote by: Daniel L. Marino (marinodl@vcu.edu)
    properties ( Access = private )
        x
        n
    end
    methods 
        
        function obj = PowNode(x, n)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            obj.n = n;
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            obj.x = x;
            y = x.^obj.n;
        end
               
        function dl_dx = backward_inputs(obj, dl_dy)
            dl_dx = dl_dy .* obj.n.*obj.x.^(obj.n-1);
        end
        
        
    end
    
end