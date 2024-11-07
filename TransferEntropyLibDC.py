import numpy as np
from scipy import stats
from entropyLibDC import *

def Conditional_MI(ytp1, xttau, yttau, n=16, buffer_coefficient=0.3, e=1e-12):
    """
    Calculate the conditional mutual information I(ytp1; xttau | yttau).

    Parameters:
    ytp1 : array_like 
        Future state of sequence y. => y_{t+1}
    xttau : array_like 
        Past states of sequence x. => x_t^tau
    yttau : array_like
        Past states of sequence y. => y_t^tau
    n : int, optional
        Number of Gauss points for numerical integration.
    buffer_coefficient : float, optional
        Coefficient to determine the buffer around the range of data for KDE.
    e : float, optional
        Small constant to prevent log of zero.

    Returns:
    float
        The calculated conditional mutual information.
    """

    # Concatenate arrays for joint distributions and create KDEs
    big_joint = np.hstack([ytp1, xttau, yttau])
    small_j1 = np.hstack([ytp1, yttau])
    small_j2 = np.hstack([xttau, yttau])
    kde_bj = stats.gaussian_kde(big_joint.T)
    kdey_tau = stats.gaussian_kde(yttau.T)
    kde_s1 = stats.gaussian_kde(small_j1.T)
    kde_s2 = stats.gaussian_kde(small_j2.T)
    # Calculate Gauss points and weights
    upper, lower = big_joint.max(0), big_joint.min(0)
    buffer = buffer_coefficient * (upper - lower)
    # Assuming GLJointNW_general is defined elsewhere
    glpw, _ = GLJointNW_general(N=[n] * big_joint.shape[-1], Lower=lower-buffer, Upper=upper+buffer)

    # Extract points and weights for each distribution
    d_ytp1, d_xtau, d_ytau = [ytp1.shape[-1], xttau.shape[-1], yttau.shape[-1]]
    gauss_bigjoint = glpw[:, :-1]
    gaussp_ytau = glpw[:, d_ytp1+d_xtau:d_ytp1+d_xtau+d_ytau] # y_t^tau
    gauss_sj1 = np.hstack([glpw[:, :d_ytp1], gaussp_ytau]) # [y_{t+1},y_t^tau]
    gauss_sj2 = np.hstack([glpw[:, d_ytp1:d_ytp1+d_xtau], gaussp_ytau]) #[x_t^tau, y_t^tau]


    w = glpw[:, -1]  # Gauss weights
    # Evaluate KDEs at the Gauss points
    pbj = kde_bj(gauss_bigjoint.T)
    pytau = kdey_tau(gaussp_ytau.T)
    ps1 = kde_s1(gauss_sj1.T)
    ps2 = kde_s2(gauss_sj2.T)
    # Regularize probabilities
    pbj, pytau, ps1, ps2 = [np.maximum(arr, e) for arr in [pbj, pytau, ps1, ps2]]
    
    # Compute and return the conditional mutual information
    conditional_MI = (pbj * (np.log2(pytau) + np.log2(pbj) - np.log2(ps1) - np.log2(ps2)) * w).sum()
    return conditional_MI

def TEttau(source,target,t,tau,n=16):
    """
    Calculates the transfer entropy from the source sequence to the target sequence at a specific observed time step `t`,
    considering a past history of length `tau`. This function uses a numerical integration method with `n` Gauss points.

    Inputs:
      - source, target: Two arrays of shape (N, T), where N is the number of observations and T is the number of time steps.
      - t: The current observed time step for which the transfer entropy is being calculated.
      - tau: The length of past history to consider for calculating transfer entropy.
      - n: The number of Gauss points used for numerical integration in the entropy calculation.

    Output:
      - TEint: A scalar value representing the transfer entropy from `source` to `target` at the specified time step `t`.
    """
    ytp1=target[:,[t+1]]
    yttau=target[:,t-tau+1:t+1]
    xttau=source[:,t-tau+1:t+1]
    TEint=Conditional_MI(ytp1,xttau,yttau,n=n)
    return TEint


def Transfer_Entropy_Sequence(source,target,tau=1,n=16):
    """
    Computes a sequence of transfer entropy values from the source sequence to the target sequence for a series of
    observed time steps. This calculation takes into account a specified past history length (`tau`) and performs numerical
    integration using a given number of Gauss points (`n`). This function is designed to evaluate the dynamic transfer of
    information between the two sequences over time.

    Inputs:
      - source, target: Two numpy arrays of shape (N, T), where N represents the number of observations and T the total number
        of time steps, indicating the source and target sequences, respectively.
      - tau: An integer indicating the length of the past history to consider in the transfer entropy calculation, allowing
        for the assessment of temporal dependencies over the specified length.
      - n: The number of Gauss points used for the numerical integration, impacting the accuracy and computational demand of
        the entropy calculations.

    Outputs:
      - Transfer entropy sequence: A numpy array containing the calculated transfer entropy values for each time step from `tau`
        to `T-1`, illustrating how information flow from source to target evolves over time.
      - t_interval: A numpy array of observation steps corresponding to each transfer entropy value, useful for plotting and
        analysis, indicating the time steps at which transfer entropy was computed.

    Note: The function ensures that transfer entropy values are strictly non-negative by applying a minimum threshold of 1e-12,
    avoiding numerical issues with logarithms and zero values.
    """
    T=source.shape[-1]
    transfer_entropy_values=[]
    t_interval=[]

    for t in range(tau-1,T-1):
        TEint=TEttau(source,target,t,tau,n)
        transfer_entropy_values.append(TEint)
        t_interval.append(t+1)
    transfer_entropy_values=np.array(transfer_entropy_values)
    transfer_entropy_values=np.maximum(transfer_entropy_values,1e-12)
    return transfer_entropy_values, t_interval


#############################################Permutable Testing#############################################

def random_shuffle(A):
    """
    Randomly shuffle a sequential dataset along its temporal dimension.
    
    Parameters:
    - A: numpy array. The sequential data to be shuffled.
        For a 2D array (matrix), rows represent different sequences or variables,
        and columns represent time steps (temporal dimension).
        For a 1D array, it represents a single sequence of data over time.
    
    Returns:
    - numpy array: The shuffled array with the same shape as A.
    
    The function checks the dimensionality of A to determine how to shuffle:
    - If A is 2D (matrix), it shuffles the columns (time steps) while keeping the row order (sequences).
    - If A is 1D, it shuffles the array elements.
    - If A has an unsupported shape, it prints an 'Error shape' message.
    """
    if len(A.shape) == 2:  # If A is a 2D matrix
        return A[:, np.random.permutation(A.shape[1])]
    elif len(A.shape) == 1:  # If A is a 1D array
        return A[np.random.permutation(A.shape[0])]
    else:  # If A has an unsupported shape
        print('Error: Unsupported array shape.')

def Transfer_Entropy_Sequence_plus_permutable_test(source,target,tau=1,n=16,r=50,rho=0.9):
    """
    Extending Transfer_Entropy_Sequence with a permutation test for Transfer Entropy (TE) significance analysis.
    This function evaluates the significance of TE values by comparing against a distribution of TE values
    obtained from permuted source sequences, helping to filter out TE values likely resulting from random chance.
    
    Inputs:
        - source, target: Input sequences for TE calculation.
        - tau (int): Time delay for TE calculation, default is 1.
        - n (int): Number of Gauss-Legendre points, default is 16.
        - r (int): Number of permuted sequences for significance testing.
        - rho (float): Threshold probability to filter the TE values, between 0 and 1.
    
    Outputs:
        - original_TE (numpy array): Transfer entropy sequence before permutation test.
        - TE_purify (numpy array): Adjusted Transfer entropy sequence based on permutation test results.
        - cdf_te_values (numpy array): A list containing the Cumulative Distribution Function (CDF) values of TE, in temporal order.
        - t_interest (numpy array): Array of observed steps, mainly used for plotting.
    
    Note: This function assumes the presence of a pre-defined Transfer_Entropy_Sequence function for initial TE calculations.
    """
    permuted_sources=[random_shuffle(source) for i in range(r)]
    permuted_TEseqs_list=[Transfer_Entropy_Sequence(permuted_source,target,tau=tau,n=n)[0] for permuted_source in permuted_sources]
    permuted_TEseqs=np.stack(permuted_TEseqs_list)
    original_TE,t_interest=Transfer_Entropy_Sequence(source,target=target,tau=tau,n=n)

    cdf_te_values_list=[]
    for col in range(permuted_TEseqs.shape[-1]):
        kde_col=stats.gaussian_kde(permuted_TEseqs[:,col])
        #p_te_col=kde_col(original_TE[col])
        cdf_te_values_list.append(kde_col.integrate_box_1d(-np.inf, original_TE[col]))
    cdf_te_values=np.array(cdf_te_values_list)
    TE_purify=original_TE.copy()
    TE_purify[cdf_te_values<rho]=0
    return original_TE, TE_purify,cdf_te_values,t_interest