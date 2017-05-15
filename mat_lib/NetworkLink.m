classdef NetworkLink
    properties
        src_port
        dest_port
        n_dest
    end
    
    
    methods
        function obj = NetworkPort(src, dest)
            src_node = cell(0);
            dest_node = cell(0);
            n_dest = 0;
        end
        
        function tensor = source( obj )
            tensor = src_node.outputs
        end
        
        function tensor = dest( obj )
            
        end
                
    end
    
end