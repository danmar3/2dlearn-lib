classdef NetworkInputPort < handle
    properties
        node % 
        name % name of the port's variable
        data
        dl_dx
        src_port
        waiting % true if the port is waiting for getting the data
    end
    
    
    methods
        function obj = NetworkInputPort(node, name, src_port)
            obj.node = node;
            obj.name = name;
            obj.waiting = true;
            obj.src_port = src_port;
        end
        
        function x = get_data( obj )
            x = obj.data;
        end
        
         function set_data( obj, x )
            obj.data= x;
            obj.waiting = false;
        end
        
    end
    
end