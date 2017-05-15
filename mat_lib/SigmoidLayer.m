classdef SigmoidLayer < NetworkNode
    % SigmoidLayer: layer for a feedforward network that computes sigmoid
    %               function
    % 
    % Wrote by: Daniel L. Marino (marinodl@vcu.edu)
    properties ( Access = private )
        y
    end
    methods 
        
        function obj = SigmoidLayer(x)           
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            obj.y = 1./(1+exp(-x));
            y = obj.y;
        end
        
        function backward_params(obj, de_dy)
            
        end
        
        function dl_dx = backward_inputs(obj, de_dy)
            
            dl_dx = de_dy .* obj.y .* (1 - obj.y);
            
        end
        
        
    end
    
end