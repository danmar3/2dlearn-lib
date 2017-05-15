function imshow_batch( images, M, N )
%imshow_batch plots several images
% Inputs:
%   @param images: Tensor with the images to be plot. Must be indexed by
%                  the last index
%   @param N: number of images to be arranged vertically
%   @param M: number of images to be arranged horizontally
%
% Wrote by: Daniel L. Marino (marinodl@vcu.edu)
%   Modern Heuristics Research Group (MHRG)
%   Virginia Commonwealth University (VCU), Richmond, VA
%   http://www.people.vcu.edu/~mmanic/

if ndims(images) == 3
    H = size(images, 1);
    W = size(images, 2);
    n_images = size(images, 3);
    
    assert(M*N == n_images, 'Number of images must not change while arranging them')
    
    images = reshape(images, H, W*N, M);
    images = reshape( permute(images, [1, 3, 2]),  H*M, W*N);
    
    imshow(images);    
    
elseif ndims(images) == 4
    error('Not implemented yet for more than one channel images');    
end

end

