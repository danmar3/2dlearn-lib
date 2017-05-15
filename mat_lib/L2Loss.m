classdef L2Loss < NetworkNode
    % l2 loss node
    %
    % Note: all vectors are asumed to be row vectors
    %
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        % y: output loss
        % x: input probabilities
        % x_size: size of the input
        x_size
        x
    end
    methods
        function obj = L2Loss(x)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'err'}, {'x'}, {x});
        end
        
        function err = forward(obj, x)
            % forward: forward pass, computes y=sum( -log(x) )
            %    - x: 1d vector containing the inputs. Each row correspond
            %         to a different probability. [sample, 1]
            % returns:
            %    - y: scalar. negative log likelihood. [1, 1]
            obj.x_size = size(x);
            obj.x = x;
            
            err = sum( 0.5*(obj.x-obj.const.y).^2 );
                        
        end
        
        function dl_dx = backward_inputs(obj, dl_derr)
            %   - e: represents the backpropagated Error, it also can be
            %        seen as the ordered derivative of the final function
            %        with respect the outputs of the Gmm model
            % return:
            %   - de_dx: derivative of the error with respect each given
            %            probability
            
            dl_dx = dl_derr.*(obj.x-obj.const.y);     
            
        end
        
        
    end
end