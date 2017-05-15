classdef GmmLayer < GmmModel
    % GmmLayer: layer for a feedforward network with mu, sigma and w as
    %           parameters
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        CovType
        sigma_sym
        tol % for preventing too skewed matrices 
    end
    methods 
        
        function obj = GmmLayer(n_dim, n_kernels, CovType)            
            obj = obj@GmmModel(n_dim, n_kernels);
            obj.n_inputs= 0;
            obj.n_outputs= 1;
                
            if (nargin == 2) 
                CovType = 'diagonal';
            end
            obj.CovType = CovType;
            
            if strcmp(CovType, 'diagonal')
                obj.params.mu = rand(1, n_kernels, n_dim);
                obj.params.sigma = rand(1, n_kernels, n_dim);
                obj.params.w = rand(1, n_kernels, 1);     

                obj.dparams = obj.params;
                obj.dparams.mu = rand(1, n_kernels, n_dim);
                obj.dparams.sigma = rand(1, n_kernels, n_dim);
                obj.dparams.w = rand(1, n_kernels, 1);
                
                % call node configuration method
                NodeConf(obj, {'y'});
            elseif strcmp(CovType, 'full')
                obj.params.mu = rand(1, n_kernels, n_dim);
                obj.params.sigma = rand(1, n_kernels, n_dim, n_dim);
                obj.params.w = rand(1, n_kernels, 1);     

                obj.dparams = obj.params;
                obj.dparams.mu = rand(1, n_kernels, n_dim);
                obj.dparams.sigma = rand(1, n_kernels, n_dim, n_dim);
                obj.dparams.w = rand(1, n_kernels, 1);     
                
                obj.tol = 0.01;
                
                % call node configuration method
                NodeConf(obj, {'y'});
            elseif strcmp(CovType, 'constrained')
                obj.n_outputs= 2;
                obj.params.mu = rand(1, n_kernels, n_dim);
                obj.params.q = repmat(permute(eye( n_dim ),[4,3,1,2]), [1, n_kernels, 1, 1]);
                obj.params.d = rand(1, n_kernels, n_dim);
                obj.params.w = rand(1, n_kernels, 1);     

                obj.dparams = obj.params;
                obj.dparams.mu = zeros(1, n_kernels, n_dim);
                obj.dparams.q = zeros(1, n_kernels, n_dim, n_dim);
                obj.dparams.d = zeros(1, n_kernels, n_dim);
                obj.dparams.w = zeros(1, n_kernels, 1); 
                
                obj.tol = 0.001;
                % call node configuration method
                NodeConf(obj, {'y', 'q'});
            end
            
        end
        
        function [varargout] = forward(obj)
            if strcmp(obj.CovType, 'diagonal')
                varargout{1} = forward@GmmModel(obj, obj.params.mu, obj.params.sigma, obj.params.w, obj.const.x);
                
            elseif strcmp(obj.CovType, 'full')
                % ensure sigma is symmetric
                obj.sigma_sym = obj.params.sigma + permute(obj.params.sigma, [1,2,4,3]);
                
                sigma_pd = zeros(size(obj.sigma_sym));
                % ensure sigma is p.d.
                for k=1:obj.n_kernels
                    sigma_sym_k = permute(obj.sigma_sym(:,k,:,:),[1,3,4,2]);
                    sigma_pd(:,k,:,:) = permute(batch_matmul(sigma_sym_k, sigma_sym_k), [1, 4, 2, 3]);
                end  
                
                % add small constant to prevent sigma to be too skew
                sigma_pd = bsxfun( @plus, sigma_pd, permute(obj.tol*eye(obj.n_dim), [4, 3, 1, 2]) );
                
                
                varargout{1} = forward@GmmModel(obj, obj.params.mu, sigma_pd, obj.params.w, obj.const.x);
                
            elseif strcmp(obj.CovType, 'constrained')
                % ensure s is positive
                d_pd = exp(obj.params.d);
                
                sigma_pd = zeros(size(obj.params.q));
                d_pd_q =  bsxfun(@times, d_pd, permute(obj.params.q, [1,2,4,3])); % diag(d_pd)*Q'
                % compute sigma
                for k=1:obj.n_kernels
                    q_k = permute(obj.params.q(:,k,:,:),[1,3,4,2]);
                    d_pd_q_k = permute(d_pd_q(:,k,:,:),[1,3,4,2]);
                    sigma_pd(:,k,:,:) = permute(batch_matmul(q_k, d_pd_q_k), [1, 4, 2, 3]);
                end  
                
                % add small constant to prevent sigma to be too skew
                sigma_pd = bsxfun( @plus, sigma_pd, permute(obj.tol*eye(obj.n_dim), [4, 3, 1, 2]) );
                                
                varargout{1} = forward@GmmModel(obj, obj.params.mu, sigma_pd, obj.params.w, obj.const.x);
                varargout{2} = obj.params.q;
            end
        end
        
        function dparams = backward_params(obj, dl_dy, dl_dq)
            if strcmp(obj.CovType, 'diagonal')
                [obj.dparams.mu, obj.dparams.sigma, obj.dparams.w] = backward(obj, dl_dy);
                
            elseif strcmp(obj.CovType, 'full')
                [obj.dparams.mu, dl_dsigma_sym, obj.dparams.w] = backward(obj, dl_dy);
                
                dl_dsigma = zeros(size(dl_dsigma_sym)); % TODO: change this
                % account for p.d. transform
                for k=1:obj.n_kernels
                    sigma_sym_k = permute(obj.sigma_sym(:,k,:,:),[1,3,4,2]);
                    dl_dsigma_sym_k = permute(dl_dsigma_sym(:,k,:,:),[1,3,4,2]);
                    dl_dsigma_1 = permute(batch_matmul(dl_dsigma_sym_k, sigma_sym_k), [1, 4, 2, 3]);
                    dl_dsigma_2 = permute(batch_matmul(sigma_sym_k, dl_dsigma_sym_k), [1, 4, 2, 3]);
                    dl_dsigma(:,k,:,:) = dl_dsigma_1 + dl_dsigma_2;
                    
                end  
                %dl_dsigma = 2*dl_dsigma;
                % account for symmetric transform
                obj.dparams.sigma= dl_dsigma + permute(dl_dsigma, [1,2,4,3]);
                
            elseif strcmp(obj.CovType, 'constrained')
                [obj.dparams.mu, dl_dsigma, obj.dparams.w] = backward(obj, dl_dy);
                
                % dl_dd, dl_dq
                for i=1:size(dl_dsigma, 1)
                    for k=1:obj.n_kernels
                        d_ik = squeeze(obj.params.d(i,k,:));
                        dl_dsigma_ik = squeeze(dl_dsigma(i,k,:,:));
                        q_ik = squeeze(obj.params.q(i,k,:,:));
                        
                        obj.dparams.d(i,k,:) = diag(q_ik'*dl_dsigma_ik'*q_ik).*exp(d_ik);
                        
                        obj.dparams.q(i,k,:,:) = (dl_dsigma_ik + dl_dsigma_ik')*q_ik*diag(exp(d_ik));
                    end
                end
                
                % 
                obj.dparams.q = obj.dparams.q + dl_dq;
                
            end
                        
            dparams= obj.dparams;
        end
        
        function backward_inputs(obj)
            
        end
        
        
    end
    
end