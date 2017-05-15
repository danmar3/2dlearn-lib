function bdims = broadcasted_dims(y, x1)
%broadcasted_dims: returns the dimentions that were broadcasted in the
%                  operation y = bsxfun(op, x1, x2). It is assumed that the
%                  operators dimentions are valid
x1_size = [size(x1) ones(1, ndims(y)-ndims(x1))];
bdims = find(size(y) - x1_size);

end

