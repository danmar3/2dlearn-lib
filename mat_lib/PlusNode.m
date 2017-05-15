classdef PlusNode < NetworkNode
    % PlusNode: element-wise sum between x1 and x2. Broadcast enabled
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        bdims_x1
        bdims_x2
    end
    
    methods 
        
        function obj = PlusNode(x1, x2)
            obj.n_inputs= 2;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x1','x2'}, {x1, x2});
        end
        
        function y = forward(obj, x1, x2)
            y = bsxfun( @plus, x1 , x2);
            obj.bdims_x1 = broadcasted_dims(y, x1);
            obj.bdims_x2 = broadcasted_dims(y, x2);
        end
        
        function [dl_dx1, dl_dx2] = backward_inputs(obj, de_dy)
            dl_dx1 = de_dy;
            for i= obj.bdims_x1
                dl_dx1 = sum(dl_dx1, i);
            end
            
            dl_dx2 = de_dy;
            for i= obj.bdims_x2
                dl_dx2 = sum(dl_dx2, i);
            end
        end
        
    end    
end