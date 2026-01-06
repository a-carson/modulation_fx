from numba import jit
import numpy as np
import numpy.typing as npt

@jit(nopython=True)
def state_space_biquad(b: npt.NDArray, a: npt.NDArray, x: npt.NDArray):
    num_samples = x.shape[-1]
    y = np.zeros_like(x)

    b = b / a[0]
    a = a / a[0]
    A = np.array([[-a[1], -a[2]], [1.0, 0]], dtype=x.dtype)
    B = np.array([[1.0, 0]], dtype=x.dtype).T
    C = np.array([b[1] - b[0]*a[1], b[2] - b[0]*a[2]], dtype=x.dtype)
    D = np.array([b[0]], dtype=x.dtype)
    state = np.zeros((2, 1), dtype=x.dtype)
    for n in range(num_samples):
        state, yn = state_space_biquad_update(A, B, C, D, x[n], state)
        y[n] = yn[0]

    return y

@jit(nopython=True)
def state_space_biquad_update(A: npt.NDArray, B: npt.NDArray, C: npt.NDArray, D: npt.NDArray, x: npt.NDArray, state: npt.NDArray):
    yn = C @ state + D * x
    state = A @ state + B * x
    return state, yn

@jit(nopython=True)
def biquad_update(b: npt.NDArray, a: npt.NDArray, h1: npt.NDArray, h2: npt.NDArray, x: npt.NDArray):
    y = (b[1] - b[0] * a[1]) * h1 + (b[2] - b[0]*a[2]) * h2 + b[0] * x
    h1_next = -a[1] * h1 - a[2] * h2 + x
    h2_next = h1
    return y, h1_next, h2_next

@jit(nopython=True)
def svf_filter(svf_params: npt.NDArray, x: npt.NDArray):
    '''
    SVF filter implementation

    :param svf_params: in the form [f, R, mLP, mBP, mHP]
    :param x: input signal
    :return:
    '''
    num_samples = x.shape[-1]
    y = np.zeros_like(x)
    h2 = np.zeros(1)
    h1 = np.zeros(1)
    g, R, m_lp, m_bp, m_hp = np.split(svf_params, 5)
    for n in range(num_samples):
        yn, h1, h2 = svf_state_update(x[n], h1, h2, g, R, m_lp, m_bp, m_hp)
        y[n] = yn[0]

    return y

@jit(nopython=True)
def svf_state_update(x: npt.NDArray,
                     h1: npt.NDArray,
                     h2: npt.NDArray,
                     g: npt.NDArray,
                     R: npt.NDArray,
                     m_lp: npt.NDArray,
                     m_bp: npt.NDArray,
                     m_hp: npt.NDArray):
    '''
    SVF state update
    See: https://www.dafx.de/paper-archive/2020/proceedings/papers/DAFx2020_paper_52.pdf

    :param x:
    :param h1: state
    :param h2: state
    :param g: frequency
    :param R: resonance
    :param m_lp: lowpass mix
    :param m_bp: bandpass mix
    :param m_hp: highpass mix
    :return:
    '''

    y_bp = (g * (x - h2) + h1) / (1 + g * (g + 2 * R))
    y_lp = g * y_bp + h2
    y_hp = x - y_lp - 2 * R * y_bp
    h1 = 2 * y_bp - h1
    h2 = 2 * y_lp - h2
    y = m_lp * y_lp + m_bp * y_bp + m_hp * y_hp

    return y, h1, h2


@jit(nopython=True)
def time_varying_comb(x, delay, b: npt.NDArray, a: npt.NDArray):
    '''
    Combined feedforward and feedback comb filter structure
    See: https://ccrma.stanford.edu/~jos/pasp/Allpass_Two_Combs.html

    :param x: input signal
    :param delay: time-varying delay signal
    :param b: feedforward coefficient
    :param a: feedbackcoefficien
    :return: filtered signal
    '''
    max_delay = int(np.ceil(np.max(delay))) + 1
    buff = np.zeros(max_delay + 1)
    num_samples = x.shape[-1]
    delay = np.clip(delay, a_min=0, a_max=None)
    y = np.zeros_like(x)
    for n in range(num_samples):
        delay_ex = delay[n]
        delay_floor = int(np.floor(delay_ex))
        delay_ceil = int(np.ceil(delay_ex))
        alpha = delay_ex - delay_floor
        vn_D = (1 - alpha) * buff[delay_floor] + alpha * buff[delay_ceil]
        vn = x[n] + a * vn_D
        yn = b[0] * vn + b[1] * vn_D
        buff[0] = vn
        y[n] = yn
        buff = np.roll(buff, 1)
    return y

@jit(nopython=True)
def time_varying_svf_comb(x, delay, b: npt.NDArray, a: npt.NDArray, svf_params: npt.NDArray):
    '''
    Combined feedforward and feedback comb filter structure, where the delay line includes an SVF filter in series
    See: https://ccrma.stanford.edu/~jos/pasp/Allpass_Two_Combs.html

    :param x: input signal
    :param delay: time-varying delay signal
    :param b: feedforward coefficient
    :param a: feedback coefficient
    :return: filtered signal
    '''
    max_delay = int(np.ceil(np.max(delay))) + 1
    buff = np.zeros(max_delay + 1)
    num_samples = x.shape[-1]
    delay = np.clip(delay, a_min=0, a_max=None)
    y = np.zeros_like(x)
    h2 = np.zeros(1)
    h1 = np.zeros(1)
    g, R, m_lp, m_bp, m_hp = np.split(svf_params, 5)
    for n in range(num_samples):
        delay_ex = delay[n]
        delay_floor = int(np.floor(delay_ex))
        delay_ceil = int(np.ceil(delay_ex))
        alpha = delay_ex - delay_floor
        vn_D = (1 - alpha) * buff[delay_floor] + alpha * buff[delay_ceil]
        vn = x[n] + a * vn_D
        vn_D, h1, h2 = svf_state_update(vn_D, h1, h2, g, R, m_lp, m_bp, m_hp)
        yn = b[0] * vn + b[1] * vn_D
        buff[0] = vn
        y[n] = yn[0]
        buff = np.roll(buff, 1)
    return y

@jit(nopython=True)
def time_varying_bq_comb(x, delay, b: npt.NDArray, a: npt.NDArray, bq_b: npt.NDArray, bq_a: npt.NDArray):
    '''
    Combined feedforward and feedback comb filter structure, where the delay line includes an SVF filter in series
    See: https://ccrma.stanford.edu/~jos/pasp/Allpass_Two_Combs.html

    :param x: input signal
    :param delay: time-varying delay signal
    :param b: feedforward coefficient
    :param a: feedback coefficient
    :return: filtered signal
    '''
    max_delay = int(np.ceil(np.max(delay))) + 1
    buff = np.zeros(max_delay + 1)
    num_samples = x.shape[-1]
    delay = np.clip(delay, a_min=0, a_max=None)
    y = np.zeros_like(x)
    # biquad coeffs
    bq_b = bq_b / bq_a[0]
    bq_a = bq_a / bq_a[0]
    h2 = np.zeros(1)
    h1 = np.zeros(1)
    for n in range(num_samples):
        delay_ex = delay[n]
        delay_floor = int(np.floor(delay_ex))
        delay_ceil = int(np.ceil(delay_ex))
        alpha = delay_ex - delay_floor
        vn_D = (1 - alpha) * buff[delay_floor] + alpha * buff[delay_ceil]
        vn = x[n] + a * vn_D
        vn_D, h1, h2 = biquad_update(bq_b, bq_a, h1, h2, vn_D)
        yn = b[0] * vn + b[1] * vn_D
        buff[0] = vn[0]
        y[n] = yn[0]
        buff = np.roll(buff, 1)
    return y

#@jit(nopython=True)
def time_varying_bq2_comb(x, delay, b: npt.NDArray, a: npt.NDArray, bq_b: npt.NDArray, bq_a: npt.NDArray):
    '''
    Combined feedforward and feedback comb filter structure, where the delay line includes an SVF filter in series
    See: https://ccrma.stanford.edu/~jos/pasp/Allpass_Two_Combs.html

    :param x: input signal
    :param delay: time-varying delay signal
    :param b: feedforward coefficient
    :param a: feedback coefficient
    :return: filtered signal
    '''
    max_delay = int(np.ceil(np.max(delay))) + 1
    buff = np.zeros(max_delay + 1)
    num_samples = x.shape[-1]
    delay = np.clip(delay, a_min=0, a_max=None)
    y = np.zeros_like(x)
    # biquad coeffs
    bq_b = bq_b / bq_a[0]
    bq_a = bq_a / bq_a[0]
    h2 = np.zeros(1)
    h1 = np.zeros(1)
    for n in range(num_samples):
        delay_ex = delay[n]
        delay_floor = int(np.floor(delay_ex))
        delay_ceil = int(np.ceil(delay_ex))
        alpha = delay_ex - delay_floor
        vn_D = (1 - alpha) * buff[delay_floor] + alpha * buff[delay_ceil]
        vn_D, h1, h2 = biquad_update(bq_b, bq_a, h1, h2, vn_D)
        vn = x[n] + a * vn_D
        yn = b[0] * vn + b[1] * vn_D
        buff[0] = vn
        y[n] = yn[0]
        buff = np.roll(buff, 1)
    return y

@jit(nopython=True)
def phaser_with_unit_delay_feedback(x, pole, b: npt.NDArray, a: npt.NDArray, K: int, ):
    '''

    :param x: input signal
    :param pole: time-varying pole location of the all-passes
    :param b: feedforward coefficient
    :param a: feedback coefficient
    :return: filtered signal
    '''
    num_samples = x.shape[-1]
    y = np.zeros_like(x)

    # all-pass states
    h = np.zeros(K)
    vn_D = np.zeros(1)


    for n in range(num_samples):

        # model with extra delay in FB path
        vn = x[n] + a * vn_D
        vn_D, h = cascaded_apf_state_update(h, vn, pole[n], K)
        yn = b[0] * vn + b[1] * vn_D
        y[n] = yn[0]

    return y

@jit(nopython=True)
def phaser_with_feedback(x, pole, b: npt.NDArray, a: npt.NDArray, K: int, gamma: float):
    '''

    solves:
    Ax[n] = Bx[n-1] + cu[n]

    :param x: input signal
    :param pole: time-varying pole location of the all-passes
    :param b: feedforward coefficient
    :param a: feedback coefficient
    :return: filtered signal
    '''
    num_samples = x.shape[-1]
    v = np.zeros_like(x)
    ap_out = np.zeros_like(x)

    # pre-compute coefficients
    state = np.zeros((K+1, 1))

    # state space matrices
    mat_a_temp = np.identity(K+1)
    mat_a_temp[0, -1] = - (1 - gamma) * a[0]
    mat_b_temp = np.diag(-1 * np.ones(K), -1)
    mat_b_temp[0, -1] = gamma * a[0]
    c = np.concatenate((np.ones((1, 1)), np.zeros((K, 1))), axis=0)

    for n in range(num_samples):
        p = pole[n]

        # update matrices
        mat_a = mat_a_temp + np.diag(-p * np.ones(K), -1)
        mat_b = mat_b_temp + np.diag(np.concatenate((np.zeros(1), p * np.ones(K))), 0)

        # state update
        rhs = mat_b @ state + c * x[n]
        state = np.linalg.solve(mat_a, rhs)
        ap_out[n] = b[1] * state[-1, 0]

        # comb filter update
        #vn = x[n] + state[-1, 0] * a[0]
        vn = state[0, 0]
        v[n] = b[0] * vn

    y = v + ap_out

    return y

@jit(nopython=True)
def phaser_with_feedback_and_bq1(x, pole, b: npt.NDArray, a: npt.NDArray, bq_b: npt.NDArray, bq_a: npt.NDArray, K: int):
    '''

    solves:
    Ax[n] = Bx[n-1] + cu[n]

    :param x: input signal
    :param pole: time-varying pole location of the all-passes
    :param b: feedforward coefficient
    :param a: feedback coefficient
    :return: filtered signal
    '''
    num_samples = x.shape[-1]
    v = np.zeros_like(x)
    ap_out = np.zeros_like(x)

    # pre-compute coefficients
    state = np.zeros((K+1, 1))

    # biquad coeffs
    bq_b = bq_b / bq_a[0]
    bq_a = bq_a / bq_a[0]
    h2 = np.zeros(1)
    h1 = np.zeros(1)

    # state space matrices
    mat_a_temp = np.identity(K+1)
    mat_a_temp[0, -1] = -a[0]
    mat_b_temp = np.diag(-1 * np.ones(K), -1)
    c = np.concatenate((np.ones((1, 1)), np.zeros((K, 1))), axis=0)

    for n in range(num_samples):
        p = pole[n]

        # update matrices
        mat_a = mat_a_temp + np.diag(-p * np.ones(K), -1)
        mat_b = mat_b_temp + np.diag(np.concatenate((np.zeros(1), p * np.ones(K))), 0)

        # state update
        rhs = mat_b @ state + c * x[n]
        if np.isnan(rhs).any() or np.isinf(rhs).any():
            break
        state = np.linalg.solve(mat_a, rhs)

        ap_out_n, h1, h2 = biquad_update(bq_b, bq_a, h1, h2, state[-1, 0])
        ap_out[n] = b[1] * ap_out_n[0]

        # comb filter update
        vn = x[n] + state[-1, 0] * a[0]
        v[n] = b[0] * vn

    y = v + ap_out

    return y

@jit(nopython=True)
def phaser_with_feedback_and_bq2(x, pole, b: npt.NDArray, a: npt.NDArray, bq_b: npt.NDArray, bq_a: npt.NDArray, K: int):
    '''

    :param x: input signal
    :param pole: time-varying pole location of the all-passes
    :param b: feedforward coefficient
    :param a: feedback coefficient
    :return: filtered signal
    '''
    num_samples = x.shape[-1]
    y = np.zeros_like(x)


    # pre-compute coefficients
    bq_b /= bq_a[0]
    bq_a /= bq_a[0]
    alpha_0 = bq_b[1] - bq_b[0] * bq_a[1]
    alpha_1 = bq_b[2] - bq_b[0] * bq_a[2]

    mat_a_temp = np.identity(K+4)
    mat_a_temp[0, -1] = -a[0]
    mat_a_temp[-3, -4] = -1.0
    mat_a_temp[-1, -4] = -bq_b[0]

    mat_b_temp = np.diag(np.concatenate((-np.ones(K), np.zeros(1), np.ones(1), alpha_1 * np.ones(1))), -1)
    mat_b_temp[0, 0] = 0
    mat_b_temp[-1, -3] = alpha_0
    mat_b_temp[-3, -2] = -bq_a[2]

    state = np.zeros((K+4, 1))
    c = np.concatenate((np.ones((1, 1)), np.zeros((K+3, 1))), axis=0)

    for n in range(num_samples):

        p = pole[n]

        # matrices update
        diag_a = np.concatenate((-p * np.ones(K), np.zeros(3)))
        mat_a = mat_a_temp + np.diag(diag_a, -1)

        diag_b = np.concatenate((np.zeros(1), p * np.ones(K), -bq_a[1] * np.ones(1), np.zeros(2)))
        mat_b = mat_b_temp + np.diag(diag_b, 0)

        rhs = mat_b @ state + c * x[n]
        if np.isnan(rhs).any() or np.isinf(rhs).any():
            break

        state = np.linalg.solve(mat_a, rhs)
        vn = x[n] + state[-1, 0] * a[0]
        y[n] = b[0] * vn + b[1] * state[-1, 0]


    return y

@jit(nopython=True)
def time_varying_apf(a: npt.NDArray, x: npt.NDArray, K: int = 1):
    '''

    :param a: allpass coefficient
    :param x: input signal
    :param K: number of all-passes
    :return:
    '''
    num_samples = x.shape[-1]
    y = np.zeros_like(x)
    h = np.zeros(K)
    for n in range(num_samples):
        yn, h = cascaded_apf_state_update(h, x[n], a[n], K)
        y[n] = yn

    return y



@jit(nopython=True)
def apf_state_update(v: npt.NDArray, x: npt.NDArray, a: npt.NDArray):
    '''
    See https://ccrma.stanford.edu/~jos/pasp/Nested_Allpass_Filters.html
    :param h: previous state
    :param x: current input
    :param a: coefficient of TF H(z) = (a + z^{-1})/(1 + az^{-1})
    :return:
    '''

    v_next = x - a * v
    y = a * v_next + v

    return y, v_next

#@jit(nopython=True)
def cascaded_apf_state_update(v: npt.NDArray, x: npt.NDArray, a: npt.NDArray, K: int):
    '''
    See https://ccrma.stanford.edu/~jos/pasp/Nested_Allpass_Filters.html
    :param h: vector previous states
    :param x: current input
    :param a: coefficient of TF H(z) = (k + z^{-1})/(1 + kz^{-1})
    :return:
    '''

    v_next = np.zeros_like(v)

    for k in range(K):
        v_next[k] = x - a * v[k]
        x = a * v_next[k] + v[k]

    y = x
    return y, v_next
