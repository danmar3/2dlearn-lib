function options = optimoptions( solver, varargin )
if mod(length(varargin), 2)~= 0
    error(message('Input must be as (option, value) pairs'))
end
keys = varargin(1:2:end);
values = varargin(2:2:end);
options = containers.Map(keys, values);

end

