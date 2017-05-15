classdef GmmModel < NetworkNode
    % The GMM model implemented here is for an independent set of
    % parameters. i.e. mu, sigma and w are supposed to be parameters, not
    % an output from, e.g. a neural network.
    %
    % Note: all vectors are asumed to be row vectors
    %
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties
        % n_dim: number of dimentions of the model
        % n_kernels: number of kernels
        % sigma_inv: stores the value of the inverse of sigma, which is
        %            encountered several times while computing the backward
        %            pass 
        % sigma_inv_x: stores the computation of sigma_inv*(x-mu) which is
        %              encountered several times 
        % gaussians: stores the computation of each one of the gaussians
        %
        
        n_dim 
        n_kernels 
        
        sigma_inv
        sigma_inv_x_mu  
        
        w      % w: normalized weights for the gausians
        sigma  % sigma: p.d. covariance matrices of the gmm gausians
        mu     % mu: means of the gmm gausians
        
        gaussians
        
    end
    methods
        function obj = GmmModel(n_dim_in, n_kernels_in)
            obj.n_dim = n_dim_in;
            obj.n_kernels = n_kernels_in; 
            
        end
        
        function Tout = batch_inv(Tin)
            if ismatrix(Tin)
                Tout = inv(Tin);
            elseif ndims(Tin)==3
                Tout = zeros(size(Tin));
                for k=1:size(Tin,3)
                    Tout(:,:,k) = inv(Tin(:,:,k));
                end
            end
        end
        
        function y = forward(obj, mu, sigma, w, x)
            % forward: forward pass, computes y=pi'*N(x,mu,sigma)
            %          w is normalized and sigma is converted to p.d.
            %          inside the funcion
            %
            % arguments:
            %    - x: 2d tensor containing the inputs. Each row correspond
            %         to a different sample
            %    - mu: 3d tensor containing the means for the gaussians.
            %          the "depth" dimention (3rd) is used to index the
            %          gaussians.    [samples, kernel_id, dim ]
            %          TODO: change to [samples, dim, kernel_id]
            %    - sigma: 3d tensor containing the covariance matrix of the
            %             gaussians. [samples, kernel_id, dim ]
            %    - w: vector in form of a 3d tensor containing the weights
            %         for each one of the gaussians. [samples, 1 ,kernel_id]
            % returns:
            %    - y: probability p(x|mu, sigma, w) given by the gmm model.
            %         [samples, 1]
                        
            %% 1. For diagonal matrices 
            if ndims(sigma)<=3 
                y = forward_diagonal(obj, mu, sigma, w, x);
                return
            end
            
            %% 2. For non-diagonal matrices 
            
            % 2.1 transform the parameters 
            obj.mu = mu;
            obj.sigma = sigma;
            % normalize w:
            if obj.n_kernels == 1
                w= ones(size(w,1), 1);
            else
                w= exp(w); % modifier
                w= bsxfun(@rdivide, w, sum(w,2)); % modifier
            end
            obj.w= w;
            
            % 2.2 calculate inverse and determinant for all sigmas
            % TODO: implement this section
            n_gmms = size(obj.sigma,1);
            n_samples = size(x,1);
            
            obj.sigma_inv = zeros(n_gmms, obj.n_kernels, obj.n_dim, obj.n_dim);
            norm_const= zeros(n_gmms, obj.n_kernels); % normalization constant for each gaussian
            for i=1:n_gmms
                for k=1:obj.n_kernels
                    obj.sigma_inv(i,k,:,:) = inv(squeeze(sigma(i,k,:,:)));
                    norm_const(i,k) = 1/sqrt( ((2*pi)^obj.n_kernels) * det(squeeze(sigma(i,k,:,:))) );
                end
            end
            % 2.3 compute (x-mu)
            x_mu = bsxfun(@minus, permute(x, [1, 3, 2]), obj.mu);
            
            % compute sigma_inv*(x-mu), Note: it is actually calculated as
            % (x-mu)*sigma_inv given that x-mu is oredered by rows.
            % remember also that sigma_inv is symetric
            obj.sigma_inv_x_mu = zeros(n_samples, obj.n_kernels, obj.n_dim);
            for k=1:obj.n_kernels
                obj.sigma_inv_x_mu(:,k,:) = batch_matmul(x_mu(:,k,:), permute(obj.sigma_inv(:,k,:,:), [1, 3, 4, 2]));
            end
            
            % 2.4 compute output
            obj.gaussians = bsxfun( @times, norm_const , exp( -0.5 * sum( x_mu .* obj.sigma_inv_x_mu ,3) ));
            
            y = sum(bsxfun( @times, w , obj.gaussians), 2);
            
            
        end
        
        function y = eval_prob(obj, x)
            % eval_prob: evaluates the probability of the gmm model
            %            acording to the class atributes( obj.mu, obj.sigma
            %            and obj.w)
            
            
            % calculate inverse and determinant for each sigma of each
            % kernel: 
            norm_const= 1./sqrt( ((2*pi)^obj.n_dim) * prod(obj.sigma, 3) );
            obj.sigma_inv = 1./obj.sigma;
            
            % compute (x-mu)
            x_mu = bsxfun(@minus, permute(x, [1, 3, 2]), obj.mu);
            
            % compute sigma_inv*(x-mu), Note: it is actually calculated as
            % (x-mu)*sigma_inv given that x-mu is oredered by rows.
            % remember also that sigma_inv is symetric
            obj.sigma_inv_x_mu = bsxfun( @times, x_mu, obj.sigma_inv);
            
            
            % compute output
            obj.gaussians = bsxfun( @times, norm_const , exp( -0.5 * sum( x_mu .* obj.sigma_inv_x_mu ,3) ));
            
            y = sum(bsxfun( @times, obj.w , obj.gaussians), 2);
            
        end
        
        function y = forward_diagonal(obj, mu, sigma, w, x)
            % forward: forward pass, computes y=pi'*N(x,mu,sigma), 
            %          optimized for diagonal sigma
            %    - x: 2d tensor containing the inputs. Each row correspond
            %         to a different sample
            %    - mu: 3d tensor containing the means for the gaussians.
            %          the "depth" dimention (3rd) is used to index the
            %          gaussians
            %    - sigma: 3d tensor containing the vectors (in row format)
            %             corresponding to the diagonal elements of the
            %             covariance matrix of the gaussians
            % 
            
            obj.mu = mu;
            % convert covariances into positive:
            sigma= exp(sigma);  
            obj.sigma = sigma;
            
            % normalize w:
            if obj.n_kernels == 1
                w= ones(size(w,1), 1);
            else
                w= exp(w); % modifier
                w= bsxfun(@rdivide, w, sum(w,2)); % modifier
            end
            obj.w= w;
            
            % evaluate the probability
            y = eval_prob(obj, x);
            
        end
        
        function de_dwin = de_dwin_eval(obj, de_dw)
            % de_dw: ordered derivative, format: [sample_id, kernel_id, 1]
            
            % w: weights, format: [sample_id, kernel_id, 1]
            if size(obj.w,1)==1 
                % simulate different input for different samples 
                % TODO:(this is very expensive) 
                w = repmat(obj.w, [size(de_dw,1), 1, 1]); 
            else
                w = obj.w;
            end
            
            w_diag = batch_mat_diag( w );
            dw_dwin = w_diag - batch_matmul( w, permute(w, [1, 3, 2]) );
            
            de_dwin = batch_matmul( dw_dwin, de_dw );
            
            %de_dwin = permute(de_dwin, [1,3,2]);
            %original:
            %{
            de_dwin = zeros(size(de_dw));
            
            for i=1:size(de_dw)
                w_i= squeeze(w(i,:,:)); % column vector
                de_dw_i= squeeze(de_dw(i,:,:)); % column vector
                
                de_dwin_i = (diag(w_i)-w_i*w_i')*de_dw_i;
                
                de_dwin(i,:,:)= de_dwin_i;
            end
            %}
        end
        
        
        function [de_dmu, de_dsigma, de_dw] = backward(obj, de_dy)
            %   - e: represents the backpropagated Error, it also can be
            %        seen as the ordered derivative of the final function
            %        with respect the outputs of the Gmm model
            % return:
            %   - de_dsigma  
            %   - de_dw  
            %
            
            % w_gaussians: tensor of shape: [samples, kernels, 1]
            w_gaussians =  bsxfun( @times, obj.w, obj.gaussians); % 
            %% derivative with respect mu
            % dgmm_dmu: tensor of shape: [samples, kernel, dim]
            dy_dmu = bsxfun( @times, w_gaussians , obj.sigma_inv_x_mu);  
            % de_dmu: tensor of shape: [samples, kernel, dim]
            de_dmu = bsxfun( @times, de_dy, dy_dmu); 
            
            %% derivative with respect sigma
            if ndims(obj.sigma_inv) <= 3 % if diagonal matrix
                % dgmm_dmu: tensor of shape: [samples, kernel, dim]
                dy_dsigma = bsxfun( @minus, obj.sigma_inv_x_mu.^2 , obj.sigma_inv);  
                dy_dsigma = 0.5 * bsxfun( @times, w_gaussians , dy_dsigma);  
                % de_dmu: tensor of shape: [samples, kernel, dim]
                de_dsigma = bsxfun( @times, de_dy, dy_dsigma); 
            else                         % if full matrix
                % sigma_inv_x_mu: (x-mu)' inv(sigma), 
                %                 tensor of shape [samples, kernel, dim ]
                
                % 1. compute (inv(sigma)(x-mu))'(inv(sigma)(x-mu))
                dy_dsigma = zeros( [size(obj.sigma_inv_x_mu) obj.n_dim]);
                for k=1:obj.n_kernels
                    dy_dsigma(:,k,:,:) = permute(batch_matmul(permute(obj.sigma_inv_x_mu(:,k,:),[1,3,2]), obj.sigma_inv_x_mu(:,k,:)), [1, 4, 2, 3]);
                end               
                
                % 2. compute (inv(sigma)(x-mu))'(inv(sigma)(x-mu)) - inv(sigma)
                dy_dsigma = bsxfun( @minus, dy_dsigma , obj.sigma_inv);  
                
                
                dy_dsigma = 0.5 * bsxfun( @times, w_gaussians , dy_dsigma);  
                % de_dsigma: tensor of shape: [samples, kernel, dim]
                de_dsigma = bsxfun( @times, de_dy, dy_dsigma); 
                
                
            end
            
            %% derivative with respect w
            if numel(obj.w)==1
                de_dw = 0;
            else
                de_dw = bsxfun( @times, de_dy , obj.gaussians);  
            end
            
            %--------------------------------  derivatives of modifiers -----------------------------------%            
            %% derivative with respect sigma_in
            % if diagonal matrix
            if ndims(obj.sigma_inv) <= 3
                de_dsigma= bsxfun(@times, de_dsigma, obj.sigma);
            end
            
            
            %% derivative with respect w_in
            de_dw = obj.de_dwin_eval(de_dw);
            
            
            %% if inputs are single parameters
            if size(obj.mu,1)==1 
                de_dmu = sum(de_dmu, 1);              
            end
            if size(obj.sigma_inv,1)==1 
                de_dsigma = sum(de_dsigma, 1);              
            end
            if size(obj.w,1)==1 
                de_dw = sum(de_dw, 1);              
            end
            
            
            
        end 
        
        
    end
end