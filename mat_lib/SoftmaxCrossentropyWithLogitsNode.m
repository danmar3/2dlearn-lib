classdef SoftmaxCrossentropyWithLogitsNode < NetworkNode
    % Computes the crossentropy between the softmax output and the provided
    % reference
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        n_samples
        prob
        reduce_dim
    end
    methods 
        
        function obj = SoftmaxCrossentropyWithLogitsNode(x, reduce_dim)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            if nargin==3
                obj.reduce_dim = reduce_dim;
            else
                obj.reduce_dim = 2;
            end
                                                            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
        end
        
        function y = forward(obj, x)
            obj.n_samples = size(x, 1);
            obj.prob = bsxfun(@rdivide, exp(x), sum(exp(x), obj.reduce_dim) );
            y = mean( -log(obj.prob(logical(obj.const.y))) );
        end
        
        function backward_params(obj, de_dy)
            
        end
        
        function dl_dx = backward_inputs(obj, de_dy)
            dl_dx = -de_dy.*(1/obj.n_samples).*( obj.const.y - obj.prob );
        end
        
        
    end
    
end