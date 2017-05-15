classdef ConstantNode < NetworkNode
    % ConstantNode: Node that has a constant value
    % 
    % Writted by: Daniel L. Marino (marinodl@vcu.edu)
    properties ( Access = private )
        
    end
    methods 
        
        function obj = ConstantNode()  
            obj.n_inputs= 0;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'y'});
        end
        
        function y = forward(obj)
            y = obj.const.x;
        end
        
        function feed(obj, x)
            obj.const.x = x;
        end
    end    
end