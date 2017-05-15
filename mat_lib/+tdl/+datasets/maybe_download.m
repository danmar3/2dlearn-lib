function maybe_download(filename, url)
% download content from internet if filename does not exist
% Inputs:
%   @param filename
%   @param url
%
% Outputs:
%
% Wrote by: Daniel L. Marino (marinodl@vcu.edu)
%   Modern Heuristics Research Group (MHRG)
%   Virginia Commonwealth University (VCU), Richmond, VA
%   http://www.people.vcu.edu/~mmanic/

if exist(filename, 'file') ~= 2
    fprintf('Downloading file from %s \n', url)
    websave(filename, url);
end

end

