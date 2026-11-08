"""Black-box I/Q response identification and explicitly qualified linear SE proxy."""
import math
import numpy as np
from verification.contracts.core import Invalid


def relative(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise Invalid('invalid response comparison')
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-15))


def linear_rate(W,H,noise_variance):
    """Gaussian-input, unit-symbol-power, white-noise linear receiver rate proxy.

    Not finite-QAM throughput, nor an exact model of nonlinear FP16 rounding.
    """
    W=np.asarray(W);H=np.asarray(H)
    if W.ndim!=2 or H.shape!=(W.shape[1],W.shape[0]) or not np.isfinite(W).all() or not np.isfinite(H).all() or not np.isfinite(noise_variance) or noise_variance<=0:
        raise Invalid('invalid equalizer/channel/noise dimensions')
    with np.errstate(over='raise',divide='raise',invalid='raise'):
        gain=W@H;power=np.abs(gain)**2
        signal=np.diag(power).copy();np.fill_diagonal(power,0.)
        noise=noise_variance*np.sum(np.abs(W)**2,axis=1)
        denominator=np.sum(power,axis=1)+noise
        if np.any(denominator<=0): raise Invalid('degenerate equalizer row; SINR undefined')
        sinr=signal/denominator;rate=np.log1p(sinr)/math.log(2)
    if not np.isfinite(rate).all(): raise Invalid('nonfinite rate proxy')
    return dict(sum_bits_per_s_per_hz=float(rate.sum()),sinr_per_stream=sinr.tolist(),
                rate_per_stream=rate.tolist(),filtered_noise_power=noise.tolist())


def identify(respond,H,Y,noise_variance,seed,limits):
    """Every `respond` call must replay the actual frozen program, not its oracle.

    Pack independent real and imaginary input basis columns in existing RHS slots.
    Test zero, dense/normal/scaled inputs, superposition and cross-RHS isolation.
    """
    H=np.asarray(H);Y=np.asarray(Y);m,u=H.shape;r=Y.shape[1]
    if Y.shape!=(m,r) or not np.isfinite(Y).all() or not np.isfinite(H).all(): raise Invalid('invalid probe inputs')
    calls=[]
    def run(label,y):
        x=np.asarray(respond(label,y))
        if x.shape!=(u,r) or not np.isfinite(x).all(): raise Invalid('incomplete/nonfinite actual probe output')
        calls.append(label);return x
    zero=run('zero',np.zeros_like(Y))
    real=np.zeros((u,m),complex);imag=np.zeros_like(real)
    for start in range(0,m,r):
        q=np.zeros_like(Y);size=min(r,m-start)
        for j in range(size):q[start+j,j]=1
        a=run(f'basis_real_{start}',q);b=run(f'basis_imag_{start}',1j*q)
        real[:,start:start+size]=a[:,:size];imag[:,start:start+size]=b[:,:size]
    W=(real-1j*imag)/2;V=(real+1j*imag)/2
    rng=np.random.default_rng(seed)
    z=(rng.standard_normal(Y.shape)+1j*rng.standard_normal(Y.shape))/math.sqrt(2)
    z*=max(float(np.linalg.norm(Y)),1e-15)/np.linalg.norm(z)
    payloads={'original':Y,'half':.5*Y,'double':2*Y,'negative':-Y,'dense':z,
              'sum':(Y+z)/math.sqrt(2),'isolated_rhs':np.zeros_like(Y),
              'permuted_rhs':Y[:,::-1]}
    payloads['isolated_rhs'][:,0]=Y[:,0]
    outputs={label:run(label,y) for label,y in payloads.items()}
    checks={}
    for label,y in payloads.items():
        actual=outputs[label];pred=W@y
        rows=[relative(pred[i],actual[i]) for i in range(u)]
        checks[label]=dict(linear_prediction_error=relative(pred,actual),
                           max_stream_prediction_error=max(rows),
                           widely_linear_prediction_error=relative(pred+V@y.conj(),actual))
    maximum=max(max(v['linear_prediction_error'],v['max_stream_prediction_error']) for v in checks.values())
    leak=float(np.linalg.norm(V)/max(float(np.linalg.norm(W)),1e-15))
    cross=float(np.linalg.norm(outputs['isolated_rhs'][:,1:])/max(float(np.linalg.norm(outputs['isolated_rhs'])),1e-15))
    superposition=relative(outputs['sum'],(outputs['original']+outputs['dense'])/math.sqrt(2))
    permutation=relative(outputs['permuted_rhs'],outputs['original'][:,::-1])
    offset=float(np.linalg.norm(zero))
    passed=(maximum<=limits['max_prediction_error'] and leak<=limits['max_complex_leakage'] and
            cross<=limits['max_cross_rhs_leakage'] and offset<=limits['max_zero_offset'] and
            superposition<=limits['max_prediction_error'] and permutation<=limits['max_prediction_error'])
    result=dict(status='RESPONSE_MODEL_CHECKED' if passed else 'RESPONSE_MODEL_REJECTED',
                replay_calls=len(calls),max_prediction_error=maximum,complex_leakage=leak,
                cross_rhs_leakage=cross,zero_offset_norm=offset,superposition_error=superposition,
                rhs_permutation_error=permutation,checks=checks)
    if passed:
        target=np.linalg.solve(H.conj().T@H+noise_variance*np.eye(u),H.conj().T)
        candidate=linear_rate(W,H,noise_variance);reference=linear_rate(target,H,noise_variance)
        loss=reference['sum_bits_per_s_per_hz']-candidate['sum_bits_per_s_per_hz']
        if loss < -1e-9: raise Invalid('linear receiver proxy exceeds MMSE optimum beyond roundoff')
        result.update(candidate_linear_se_proxy=candidate,mmse_linear_se_reference=reference,
                      se_loss_bits_per_s_per_hz=loss,
                      status='SE_PROXY_SAMPLE_CHECKED' if loss<=limits['max_rate_loss'] else 'SE_PROXY_QUALITY_FAILED')
    return result,dict(W=W,V=V,real_response=real,imag_response=imag,
                       **{'output_'+label:value for label,value in outputs.items()})
