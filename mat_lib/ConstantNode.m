classdef ConstantNode < NetworkNode
    % ConstantNode: Node that has a constant value
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/

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