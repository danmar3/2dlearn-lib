classdef SupervisedDataset < handle
% supervised dataset
%
% Wrote by: Daniel L. Marino (marinodl@vcu.edu)
%   Modern Heuristics Research Group (MHRG)
%   Virginia Commonwealth University (VCU), Richmond, VA
%   http://www.people.vcu.edu/~mmanic/

    properties
        x
        y
        data_ptr
        epochs_completed
        index_dim
    end
    methods
        function obj = SupervisedDataset(x, y)
            obj.x = x;
            obj.y = y;
            obj.data_ptr = 1;
            obj.epochs_completed = 0;
            
            % TODO: index_dim
            obj.index_dim = 1;
        end 
        
        function [x, y] = next_batch(obj, batch_size)
            start_p = obj.data_ptr;
            obj.data_ptr = obj.data_ptr + batch_size - 1;
            
            if obj.data_ptr > size(obj.x, obj.index_dim)
                obj.epochs_completed = obj.epochs_completed + 1;
                % Shuffle
                rand_p = randperm(size(obj.x, obj.index_dim));
                obj.x = obj.x(rand_p, :);
                obj.y = obj.y(rand_p, :);
                % Start next epoch
                start_p = 1;
                obj.data_ptr = batch_size - 1;
            end
            end_p = obj.data_ptr;
            
            x = obj.x(start_p:end_p, :);
            y = obj.y(start_p:end_p, :);
        end
    end
            
end