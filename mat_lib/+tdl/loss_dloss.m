function [loss, dloss] = loss_dloss(w, model, target)
    % forward pass
    model.vect2params(w);
    out = model.forward(target);
    loss= out{1};    
    % backward pass
    dloss = model.backward(target); 
end