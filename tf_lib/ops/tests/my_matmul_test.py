#************************************************************************
#      __   __  _    _  _____   _____
#     /  | /  || |  | ||     \ /  ___|
#    /   |/   || |__| ||    _||  |  _
#   / /|   /| ||  __  || |\ \ |  |_| |
#  /_/ |_ / |_||_|  |_||_| \_\|______|
#    
# 
#   Written by: Daniel L. Marino (marinodl@vcu.edu) (2016)
#
#   Copyright (2016) Modern Heuristics Research Group (MHRG)
#   Virginia Commonwealth University (VCU), Richmond, VA
#   http://www.people.vcu.edu/~mmanic/
#   
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#  
#   Any opinions, findings, and conclusions or recommendations expressed 
#   in this material are those of the author's(s') and do not necessarily 
#   reflect the views of any other entity.
#  
#   ***********************************************************************
#
#   Description: Test of the Matrix multiplication Implementation 
#
#   ***********************************************************************

import tensorflow as tf


my_matmul_module = tf.load_op_library('../kernels/bin/my_matmul_op.so')


with tf.Session('') as session:
    n= 10;
    m= 20;
    l= 5;
        
    a_mat = tf.Variable (tf.truncated_normal( [n,m], stddev= 0.1 ), name= 'a_mat')
    b_mat = tf.Variable (tf.truncated_normal( [m,l], stddev= 0.1 ), name= 'b_mat')
    
    my_product= my_matmul_module.my_matmul(a_mat, b_mat )
    product= tf.matmul(a_mat, b_mat)
    
    diff= tf.nn.l2_loss(product - my_product)
       
    tf.initialize_all_variables().run()
    
    print("\n\n --------------- running -----------------\n\n")
    [d_val, my_c, c] = session.run( [diff, my_product, product] );
    
    print("\n\n --------------- output -----------------\n\n")
    print("error:", d_val)
    print("output shape:", c.shape)
    print("\n")
