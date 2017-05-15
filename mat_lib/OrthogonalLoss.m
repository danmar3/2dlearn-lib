classdef OrthogonalLoss < NetworkNode
    % Negative log likelihood loss
    %
    % Note: all vectors are asumed to be row vectors
    %
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        q
    end
    methods
        function obj = OrthogonalLoss(q)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            % call node configuration method
            NodeConf(obj, {'err'}, {'q'}, {q});
        end
        
        function err = forward(obj, q)
            % forward: forward pass, computes err= Tr((QQ'-I)(QQ'-I)') 
            %    - q: 
            % returns:
            %    - err: scalar. 
            
            % dl_dd, dl_dq
            obj.q = q;
            err = 0;
            for i=1:size(q, 1)
                for k=1:size(q,2)
                    q_ik = squeeze(q(i,k,:,:));
                    A = q_ik*q_ik' - eye(size(q_ik));
                    L = A.*A;
                    err = err + sum(L(:));
                end
            end
        end
        
        function dl_dq = backward_inputs(obj, dl_derr)
            %   - e: represents the backpropagated Error, it also can be
            %        seen as the ordered derivative of the final function
            %        with respect the outputs of the loss
            % return:
            %   - de_dq: derivative of the error with respect each given
            %            matrix
            
            dl_dq = zeros(size(obj.q));
            for i=1:size(obj.q, 1)
                for k=1:size(obj.q,2)
                    q_ik = squeeze(obj.q(i,k,:,:));
                    
                    dl_dq(i,k,:,:)= 4*((q_ik*q_ik')*q_ik - q_ik);
                end
            end
            dl_dq = dl_derr*dl_dq;
        end
        
        
    end
end