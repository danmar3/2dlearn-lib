classdef FFModel < handle
    properties
        n_layers
        layers
        
        n_params
        n_params_cum
        
    end
    
    
    methods
        
        function obj = FFModel(layers)
            if nargin == 1
                obj.layers = layers;
                
                obj.setup()
            end
            
        end
        
        function setup(obj)
            obj.n_layers = length(obj.layers);
            obj.get_num_params()
        end
        
        function add_layer(obj, layer)
            % TODO
        end
        
    
        function y = forward(obj)
            if isempty(obj.n_layers)
                obj.setup()
            end
            % TODO: construct y on setup
            y = cell(obj.n_layers,1);
            
            y{1} = obj.layers{1}.forward();
            
            for l= 2:obj.n_layers
                y{l} = obj.layers{l}.forward(y{l-1});
            end
            
            % old code:
            %{
            if ~isempty(obj.layers{1}.params) && ~isempty(obj.layers{1}.const)
                params = struct2cell(obj.layers{1}.params);
                const = struct2cell(obj.layers{1}.const);

                y{1} = obj.layers{1}.forward(params{:}, const{:});

            elseif ~isempty(obj.layers{1}.params)
                params = struct2cell(obj.layers{1}.params);

                y{1} = obj.layers{1}.forward(params{:});
            end

            for l= 2:obj.n_layers % change to number of layers
                if ~isempty(obj.layers{l}.params) && ~isempty(obj.layers{l}.const)
                    params = struct2cell(obj.layers{l}.params);
                    const = struct2cell(obj.layers{l}.const);

                    y{l} = obj.layers{l}.forward(y{l-1}, params{:}, const{:});

                elseif ~isempty(obj.layers{l}.params)
                    params = struct2cell(obj.layers{l}.params);

                    y{l} = obj.layers{l}.forward(y{l-1}, params{:});

                else
                    y{l} = obj.layers{l}.forward(y{l-1});
                end

            end
            %}

        end

        function de_dw = backward(obj)
            de_dw = zeros(obj.n_params_cum(end), 1);
            idx=obj.n_params_cum(end);


            de_dinput = obj.layers{end}.backward_inputs(1);

            for l= obj.n_layers-1:-1:1
                % parameters gradient calculation
                if ~isempty(obj.layers{l}.params)
                    obj.layers{l}.backward_params(de_dinput);

                    dparams = struct2cell(obj.layers{l}.dparams);
                    dparams = cell2mat(cellfun(@(x) reshape(x,[],1), dparams, 'UniformOutput', false));

                    de_dw(idx-obj.n_params(l)+1 : idx) = dparams;
                    idx = idx - obj.n_params(l);
                end
                % inputs gradient calculation
                if obj.layers{l}.n_inputs ~= 0
                    de_dinput = obj.layers{l}.backward_inputs(de_dinput);
                end

            end

        end

        function w = params2vect(obj)

            w = zeros(obj.n_params_cum(end), 1);
            idx=1;
            for l= 1:obj.n_layers 
                if ~isempty(obj.layers{l}.params)
                    params = struct2cell(obj.layers{l}.params);
                    params = cell2mat(cellfun(@(x) reshape(x,[],1), params, 'UniformOutput', false));

                    w(idx:obj.n_params_cum(l)) = params;
                    idx = obj.n_params_cum(l);
                end
            end

        end

        function vect2params(obj, w)
            idx= 1;
            for l= 1:obj.n_layers 
                if ~isempty(obj.layers{l}.params)
                    fields = fieldnames(obj.layers{l}.params);

                    for i = 1:numel(fields)
                        param_i = obj.layers{l}.params.(fields{i});
                        obj.layers{l}.params.(fields{i}) = reshape(w(idx:idx+numel(param_i)-1), size(param_i));
                        idx = idx + numel(param_i);
                    end
                end
            end

        end

        function get_num_params(obj)
            obj.n_params= zeros(obj.n_layers, 1);

            for l= 1:obj.n_layers 
                if ~isempty(obj.layers{l}.params)
                    params = struct2cell(obj.layers{l}.params);
                    obj.n_params(l) = sum(cellfun(@numel, params));
                end
            end
            obj.n_params_cum = cumsum(obj.n_params);
        end

    end
    
    
end