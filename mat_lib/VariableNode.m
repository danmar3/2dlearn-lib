classdef VariableNode < NetworkNode
    % variable: node that stores a variable that can be tuned
    % 
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties ( Access = private )
        
    end
    
    methods 
        
        function obj = VariableNode(size_var, alpha)
            if nargin==1
                alpha = 1;
            end
            
            if length(size_var)==2
                %init_mul=1.0/sqrt(size_var(1));
                %init_mul=sqrt(6)/sqrt(size_var(1)+size_var(2));
                sigma=sqrt((alpha^2)/(size_var(1)*size_var(2)));
                obj.params.data= normrnd(0, sigma, size_var(1), size_var(2));
            else
                obj.params.data= alpha*(randn(size_var));
            end
            
            obj.n_inputs= 0;
            obj.n_outputs= 1;
            
            %;
            obj.dparams.data = zeros(size_var);
            
            % call node configuration method
            NodeConf(obj, {'y'});
        end
        
        function y = forward(obj)            
            y = obj.params.data;
        end
        
        function dparams= backward_params(obj, dl_dy)
            % dl_dy: format: [sample_id, out_id]          
            obj.dparams.data = dl_dy;
                                    
            dparams = obj.dparams;
        end
        
        function data = get_data(obj)
            data = obj.params.data;
        end
        
        function feed(obj, data)
            obj.params.data = data;
        end
    end
    
end