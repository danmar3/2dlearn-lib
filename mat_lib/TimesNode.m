classdef TimesNode < NetworkNode
    % TimesNode: compute the element-wise product between x1 and x2.
    % Broadcast enabled
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
        function obj = TimesNode(x1, x2)
            obj.n_inputs= 2;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x1','x2'}, {x1, x2});
        end
        
        function y = forward(obj, x1, x2)
            obj.x1 = x1;
            obj.x2 = x2;
            
            y = bsxfun( @times, obj.x1 , obj.x2);
        end
        
        function [dl_dx1, dl_dx2] = backward_inputs(obj, dl_dy)
            dl_dx1 = bsxfun( @times, dl_dy , obj.x2);
            for i= broadcasted_dims(dl_dy, obj.x1)
                dl_dx1 = sum(dl_dx1, i);
            end
            
            dl_dx2 = bsxfun( @times, dl_dy, obj.x1);
            for i= broadcasted_dims(dl_dy, obj.x2)
                dl_dx2 = sum(dl_dx2, i);
            end            
        end       
    end    
end