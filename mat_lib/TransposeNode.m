classdef TransposeNode < NetworkNode
    % TransposeNode: Permutes the dimensions according to perm.
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        perm
        dperm
    end
    
    methods 
        function obj = TransposeNode(x, perm)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            if nargin==1
                perm = [2 1];
            end
            obj.perm = perm;
            % dperm
            obj.dperm = zeros(size(perm));
            for i=1:length(perm)
                obj.dperm(perm(i))= i;
            end
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            y = permute(x, obj.perm);
        end
        
        function dl_dx = backward_inputs(obj, dl_dy)
            dl_dx = permute(dl_dy, obj.dperm);
        end       
    end    
end