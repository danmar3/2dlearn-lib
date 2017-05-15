classdef ReluLayer < NetworkNode
    % SigmoidLayer: layer for a feedforward network that calculates sigmoid
    %               function
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        dinput
    end
    methods 
        
        function obj = ReluLayer(x)
            % x = input port from parent node
            
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            mask = gt(x, 0);
            
            y = mask .* x;
                        
            % mask is also the derivative, so we store it to use it later
            obj.dinput = mask;
            
        end
        function backward_params(obj, de_dy)
            
        end
        
        function dl_dx = backward_inputs(obj, dl_dy)
            
            dl_dx = dl_dy .* obj.dinput; % obj.dinput stores the mask (see forward function)
            
        end
        
        
    end
    
end