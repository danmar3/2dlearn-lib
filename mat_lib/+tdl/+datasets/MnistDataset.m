classdef MnistDataset
% mnist dataset
%
% Wrote by: Daniel L. Marino (marinodl@vcu.edu)
%   Modern Heuristics Research Group (MHRG)
%   Virginia Commonwealth University (VCU), Richmond, VA
%   http://www.people.vcu.edu/~mmanic/
    
    properties
        train
        test        
    end
    
    methods
        function obj = MnistDataset(shuffle)
            % download the dataset
            if exist('Data', 'file')~=7
                mkdir('Data')
            end
            
            tdl.datasets.maybe_download(fullfile('Data', 'data_mnist_train_x.gz'), ...
                'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz');
            tdl.datasets.maybe_download(fullfile('Data', 'data_mnist_train_y.gz'), ...
                'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz');            
            tdl.datasets.maybe_download(fullfile('Data', 'data_mnist_test_x.gz'), ...
                'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz');
            tdl.datasets.maybe_download(fullfile('Data', 'data_mnist_test_y.gz'), ...
                'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz');
            
            % uncompress
            gunzip(fullfile('Data', 'data_mnist_train_x.gz'));
            gunzip(fullfile('Data', 'data_mnist_train_y.gz'));
            gunzip(fullfile('Data', 'data_mnist_test_x.gz'));
            gunzip(fullfile('Data', 'data_mnist_test_y.gz'));
            
            % load data into matlab
            file_id = fopen(fullfile('Data', 'data_mnist_train_x'), 'r');
            header = fread(file_id, 4,'uint');
            train_x = fread(file_id, 60000*28*28 ,'uint8')/255; 
            train_x = reshape(train_x, 28*28, 60000)';
            fclose(file_id);
            
            file_id = fopen(fullfile('Data', 'data_mnist_train_y'), 'r');
            header = fread(file_id, 2,'uint');
            train_y_ = fread(file_id, 60000 ,'uint8'); 
            fclose(file_id);
            
            train_y = zeros(size(train_y_, 1), 10);
            train_y(sub2ind(size(train_y), [1:size(train_y, 1)]', train_y_+1)) = 1;
            obj.train = tdl.datasets.SupervisedDataset(train_x, train_y);
            
            file_id = fopen(fullfile('Data', 'data_mnist_test_x'), 'r');
            header = fread(file_id, 4,'uint');
            test_x = fread(file_id, 10000*28*28 ,'uint8')/255; 
            test_x = reshape(test_x, 28*28, 10000)';
            fclose(file_id);
            
            file_id = fopen(fullfile('Data', 'data_mnist_test_y'), 'r');
            header = fread(file_id, 2,'uint');
            test_y_ = fread(file_id, 10000 ,'uint8'); 
            fclose(file_id);
            
            test_y = zeros(size(test_y_, 1), 10);
            test_y(sub2ind(size(test_y), [1:size(test_y, 1)]', test_y_+1)) = 1;
            obj.test = tdl.datasets.SupervisedDataset(test_x, test_y);
            
        end
    end

end

