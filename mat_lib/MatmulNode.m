classdef MatmulNode < NetworkNode
    % MatmulNode: perform matrix multiplication between x1 and x2
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        x1
        x2
    end
    
    methods 
        
        function obj = MatmulNode(x1, x2)
            obj.n_inputs= 2;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x1','x2'}, {x1, x2});
        end
        
        function y = forward(obj, x1, x2)
            obj.x1 = x1;
            obj.x2 = x2;
            
            y = obj.x1 * obj.x2;
        end
        
        function [dl_dx1, dl_dx2] = backward_inputs(obj, de_dy)
            dl_dx1 = de_dy * obj.x2';            
            dl_dx2 = obj.x1' * de_dy;            
        end        
        
    end    
end