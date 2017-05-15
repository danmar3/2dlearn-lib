classdef AffineLayer < NetworkNode
    % SigmoidLayer: layer for a feedforward network that calculates sigmoid
    %               function
    % 
    % Wrote by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        x
    end
    
    methods 
        
        function obj = AffineLayer(n_in, n_out, x)
            obj.n_inputs= 1;
            obj.n_outputs= 1;
            
            obj.params.w= rand(n_in, n_out) - 0.5;
            obj.params.b= rand(1, n_out) - 0.5;
            
            obj.dparams = obj.params;
            
            % call node configuration method
            NodeConf(obj, {'y'}, {'x'}, {x});
            
        end
        
        function y = forward(obj, x)            
            obj.x = x;
            
            y = bsxfun( @plus, obj.x*obj.params.w , obj.params.b);
                        
        end
        function dparams= backward_params(obj, dl_dy)
            % dl_dy: format: [sample_id, out_id]          
            obj.dparams.w = obj.x'*dl_dy;
            
            obj.dparams.b = sum( dl_dy , 1);         
            
            dparams = obj.dparams;
        end
        
        function dl_dx = backward_inputs(obj, de_dy)
            
            dl_dx = de_dy * obj.params.w';
            
        end
        
        
    end
    
end