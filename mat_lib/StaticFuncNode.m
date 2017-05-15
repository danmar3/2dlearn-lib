classdef StaticFuncNode < NetworkNode
    % StaticFuncNode: computes an user defined function whose gradient is
    % not back-propagated
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        func
        size_x
    end
    
    methods 
        function obj = StaticFuncNode(x, func)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            obj.func = func;
            obj.propagate_gradients = false;
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            obj.size_x = size(x);
            y = feval(obj.func, x);
        end
        
        function dl_dx = backward_inputs(obj, dl_dy)
            % TODO: instead of returning zeros, add an option to specify
            % that gradients are not propagated
            dl_dx = zeros(obj.size_x);
        end       
    end    
end