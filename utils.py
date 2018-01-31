import numpy as np
import matplotlib.pyplot as plt

RELATIVETOLL = 1e-1
ABSOLUTETOLL = 0

def plot_histograms(names, shape='All'):
    print('Plotting...')
    for name in names:
        with tf.variable_scope(name) as scope:
            tf.get_variable_scope().reuse_variables()
            try:
                weights = tf.get_variable('weights')
            except ValueError:
                continue
            weights = weights.eval()
            if shape=='All' or weights.shape[0:2] == shape:
                print(name)
                flat_weights = np.reshape(weights, [weights.shape[0], weights.shape[1], -1])
            
                if compute_sym and flat_weights.shape[0:2] == (3, 3):
                    plot_weight_symmetry(flat_weights, name)
            else:
                print('Skip...')
    print('Done.')

def plot_weight_symmetry(conv_mat_list, name):
    lenw = len(conv_mat_list)
    s_ = np.zeros((lenw,4))
    s_[:,:] = [[is_even(mat, type='HORIZONTAL'),is_even(mat, type='VERTICAL'),is_even(mat, type='DIAGSX'),is_even(mat, type='DIAGDX')] for mat in conv_mat_list]
    s_= s_.T
    as_ = np.zeros((lenw,4))
    as_[:,:] = [[is_odd(mat, type='HORIZONTAL'),is_odd(mat, type='VERTICAL'),is_odd(mat, type='DIAGSX'),is_odd(mat, type='DIAGDX')] for mat in conv_mat_list]
    as_=as_.T
    
    s_summary = np.sum(s_,axis=0)
    #more than 2 symmetry axes
    s_strong_count = 100*np.sum(s_summary >= 2)/lenw 
    
    #less than 2 symmetry axes
    s_weak_count = 100*np.sum(s_summary == 1)/lenw
    
    as_summary = np.sum(as_,axis=0)
    
    #more than 2 symmetry axes
    as_strong_count = 100*np.sum(as_summary >= 2)/lenw 
    
    #less than 2 symmetry axes
    as_weak_count = 100*np.sum(as_summary == 1)/lenw    
    
    #no symm
    none_count = 100 - s_strong_count - s_weak_count - as_strong_count - as_weak_count  
    
    
    _,ax = plt.subplots()
    ax.set_ylabel('Percentage')
    ax.set_title('Convolutional weights matrices - Symmetry')
    ax.set_xticklabels(('Multiple\nEven Symmetry', 'Single\nEven Symmetry', 'Multiple\nOdd Symmetry','Single\nOdd Symmetry', 'None'), fontsize = 5.0)
    ax.set_xticks(list(range(5)))
    plt.bar(list(range(5)),[s_strong_count,s_weak_count,as_strong_count,as_weak_count,none_count])
    plt.ylim((0, 100))
    plt.savefig(name + '.pdf')
    plt.clf()
    
    
def is_odd(mat, type='HORIZONTAL'):
    assert mat.shape == (3,3)
    rev_mat = np.copy(mat)
    rev_mat *= -1
    if type=='HORIZONTAL':
        rev_mat[1,:] = mat[1,:]
        rev_mat = np.flip(rev_mat, 0)
    if type=='VERTICAL':
        rev_mat[:,1] = mat[:,1]
        rev_mat = np.flip(rev_mat, 1)
    if type=='DIAGSX':
        rev_mat[[0,1,2],[0,1,2]] = mat[[0,1,2],[0,1,2]]
        rev_mat = rev_mat.T
    if type=='DIAGDX':
        rev_mat[[0,1,2],[2,1,0]] = mat[[0,1,2],[2,1,0]]
        rev_mat = rev_mat.T
        rev_mat = np.flip(rev_mat, 0)
        rev_mat = np.flip(rev_mat, 1) 
    return np.allclose(mat, rev_mat, RELATIVETOLL, ABSOLUTETOLL)
    
def is_even(mat, type='HORIZONTAL'):
    assert mat.shape == (3,3)
    rev_mat = np.copy(mat)
    if type=='HORIZONTAL':
        rev_mat = np.flip(rev_mat, 0)
    if type=='VERTICAL':
        rev_mat = np.flip(rev_mat, 1)
    if type=='DIAGSX':
        rev_mat = rev_mat.T
    if type=='DIAGDX':
        rev_mat = rev_mat.T
        rev_mat = np.flip(rev_mat, 0)
        rev_mat = np.flip(rev_mat, 1)
    return np.allclose(mat, rev_mat, RELATIVETOLL, ABSOLUTETOLL)

