classdef NetworkNode < handle
    % Node class of the computation graph
    %
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
    properties
        id
        
        inputs    % inputs: cell list of NetworkInputPort for the node
        outputs   % outputs: cell list of NetworkOutputPort for the node
        
        n_inputs  % n_inputs: number of inputs into the forward function, it has to be the same as the number of parents?
        n_outputs % n_outputs: number of outputs of the forward function
                
        parents % it is redundant with inputs (list of ports), but makes programming easier
        childs  % it is redundant with outputs (list of ports), but makes programming easier
        
        params   % params is a struct, provided by the user
        n_params
        const
        
        dparams
        
        propagate_gradients
        
    end
    
    
    methods
        function obj = NetworkNode()
            obj.id= -1;
            obj.n_inputs= 0;
            obj.n_outputs= 0;
            
            obj.parents= cell(0);
            obj.childs= cell(0);
            
            obj.propagate_gradients = true;
        end        
        
        function NodeConf(obj, out_names, in_names, src_inputs)
            % src_inputs is the set of NetworkOutputPorts that conect to the node's
            % input ports
            
            % configure output ports
            obj.outputs = cell(length(out_names));
            for i= 1:length(out_names)
                obj.outputs{i} = NetworkOutputPort(obj, out_names{i});
            end
            
            % configure input ports
            if nargin > 2
                obj.inputs = cell(length(in_names));
                for i= 1:length(in_names)
                    obj.inputs{i} = NetworkInputPort(obj, in_names{i}, src_inputs{i});
                    src_inputs{i}.add_dest(obj.inputs{i}); 
                end
            end           
        end
                                 
        function vect= serialize_params(obj, params_struct)
            params_cell = struct2cell(params_struct);
            
            vect = zeros(obj.n_params, 1);
                        
            % copy parameters into vector
            pointer= 1;
            for i= 1:length(params_cell)
                param_i = params_cell{i};
                numel_i = numel(param_i);
                vect(pointer:pointer+numel_i-1) = param_i(:);
                pointer = pointer + numel_i ;
            end
            
            % TODO: look for a more efficient memory usage for this
                        
        end        
        
                
        function n_params= get_num_params(obj)
            if isempty(obj.params)
                obj.n_params = 0;
            else
                params_cell = struct2cell(obj.params);
                obj.n_params = 0;
                for i= 1:length(params_cell)
                    obj.n_params = obj.n_params + numel(params_cell{i});
                end
            end
            n_params = obj.n_params;
        end       
        
        
        
        function vect= params2vect(obj)
            if isempty(obj.params)
                vect = [];
            else
                vect = obj.serialize_params(obj.params);            
            end
        end      
        
        
        
        function vect2params(obj, w) % TODO: check this for memory consumption
            if ~isempty(obj.params)
                fields = fieldnames(obj.params);
                
                pointer= 1;
                for i = 1:numel(fields)
                    param_i= obj.params.(fields{i});
                    obj.params.(fields{i}) = reshape( w(pointer:pointer+numel(param_i)-1), size(param_i) );
                    pointer = pointer + numel(param_i);
                end
                                
            end
        end        
        
        
        function reset_gradient(obj)
            % TODO(1): rewrite this
            % 1. reset derivatives of parameters
            if ~isempty(obj.dparams)
                fields = fieldnames(obj.dparams);
                
                for i = 1:numel(fields)
                    obj.dparams.(fields{i})(:) = 0;
                end
            end
            % 2. reset derivatives of input ports
            for i = 1:obj.n_inputs
                obj.inputs{i}.dl_dx = 0*obj.inputs{i}.data; %zeros(size(obj.inputs{i}.data));
            end
            
            % 3. reset derivatives of output ports
            for i = 1:obj.n_outputs
                obj.outputs{i}.dl_dy = 0*obj.outputs{i}.data; %zeros(size(obj.outputs{i}.data));
            end
                        
        end        
        
        
        function y = forward(obj) 
        end        
        
        
        
        function dl_dw = backward(obj)
        end
              
        function y = atn(obj) 
            % add to global network
            global network
            if isa(network, 'NetworkModel')
                y = network.add_node(obj);
            end
        end
                
    end
    
end