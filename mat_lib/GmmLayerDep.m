classdef GmmLayerDep < GmmModel
    % GmmLayer: layer for a feedforward network with mu, sigma and w as
    %           inputs
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties                
        CovType
        n_samples
        
    end
    methods 
        
        function obj = GmmLayerDep(n_dim, n_kernels, x, CovType)            
            obj = obj@GmmModel(n_dim, n_kernels);
            
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            if (nargin == 3) 
                CovType = 'diagonal';
            end
            obj.CovType = CovType;
            
            if strcmp(obj.CovType, 'diagonal')
                                
                % call node configuration method
                % TODO: add w, mu, sigma as outputs of the node
                NodeConf(obj, {'y'}, {'x'}, {x});
            elseif strcmp(obj.CovType, 'full')
                
                % call node configuration method
                NodeConf(obj, {'y'}, {'x'}, {x});
            elseif strcmp(obj.CovType, 'constrained')
                                
                % call node configuration method
                NodeConf(obj, {'y', 'q'}, {'x'}, {x});
            end
                
        end
        
        function [varargout] = forward(obj, x)
            % x: input that represents the parameters of the gmm. 
            %    format: [sample_id, parameters_for_current_sample]
            %    parameters_for_current_sample: [w mu sigma]
                
            n_w = obj.n_kernels;
            n_mu = obj.n_dim * obj.n_kernels;
                
            obj.n_samples= size(x,1);

            % format input
            w_in = x(:,1:n_w);

            mu_in = x(:,n_w+1:n_w+n_mu);
            mu_in = reshape(mu_in', obj.n_kernels, obj.n_dim, obj.n_samples); % transpose is to account to column-major matlab order
            mu_in = permute(mu_in, [3,1,2]);
                
                
            if strcmp(obj.CovType, 'diagonal')
                n_sigma = obj.n_dim * obj.n_kernels;
                
                sigma_in = x(:,n_w+n_mu+1:end);
                sigma_in = reshape(sigma_in', obj.n_dim, obj.n_kernels, obj.n_samples);
                sigma_in = permute(sigma_in, [3,2,1]);

                varargout{1} = forward@GmmModel(obj, mu_in, sigma_in, w_in, obj.const.y);
            end
        end
        
        function backward_params(obj, de_dy)
            
        end
        
        function dl_dx = backward_inputs(obj, dl_dy)
            [dmu, dsigma, dw] = backward(obj, dl_dy);
            
            dmu = permute(dmu, [2,3,1]);
            dmu = reshape(dmu, obj.n_dim*obj.n_kernels, obj.n_samples);
            
            if strcmp(obj.CovType, 'diagonal')
                dsigma = permute(dsigma, [2,3,1]);
                dsigma = reshape(dsigma, obj.n_dim*obj.n_kernels, obj.n_samples);

                %%%%%%
                dl_dx  = [ dw dmu' dsigma' ];
            end
            
        end
               
        
    end
    
end