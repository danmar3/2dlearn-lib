classdef NetworkInputPort < handle
    % Network input port for a node in the computation graph
    %
    % Created by: Daniel L. Marino (marinodl@vcu.edu)
    % Modern Heuristics Research Group (MHRG) 
    % Virginia Commonwealth University (VCU), Richmond, VA 
    % http://www.people.vcu.edu/~mmanic/
    
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