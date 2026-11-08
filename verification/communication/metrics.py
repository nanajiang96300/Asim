"""Paired uncoded QPSK metrics and exact channel-outage confidence bounds."""
import math
import numpy as np
from verification.contracts.core import Invalid


def cp_upper(k, n, alpha):
    """One-sided Clopper-Pearson upper bound, inverted binomial CDF.

    Trials are independent channel realizations, NOT bits sharing a channel.
    """
    if type(n) is not int or type(k) is not int or not 0 <= k <= n or n <= 0 or not 0 < alpha < 1:
        raise Invalid('invalid binomial confidence arguments')
    if k == n: return 1.
    if k == 0: return -math.expm1(math.log(alpha) / n)
    choose = [math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1) for j in range(k+1)]
    lo, hi = 0., 1.
    for _ in range(64):
        p = (lo + hi) / 2
        if not 0 < p < 1: break
        terms = [v + j*math.log(p) + (n-j)*math.log1p(-p) for j,v in enumerate(choose)]
        peak = max(terms)
        log_cdf = peak + math.log(sum(math.exp(v-peak) for v in terms))
        if log_cdf > math.log(alpha): lo = p
        else: hi = p
    return hi


def outage_interval(k, n, tail_alpha):
    return dict(lower=1-cp_upper(n-k,n,tail_alpha), upper=cp_upper(k,n,tail_alpha),
                count=k, independent_channels=n, observed_rate=k/n)


def paired(candidate, mmse, symbols):
    arrays = [np.asarray(a) for a in (candidate, mmse, symbols)]
    if any(a.shape != arrays[0].shape or a.ndim != 3 or not a.size or not np.isfinite(a).all() for a in arrays):
        raise Invalid('finite matching [batch,Nt,Nrhs] communication arrays required')
    value, reference, truth = arrays
    level = 1/math.sqrt(2)
    if not (np.allclose(np.abs(truth.real),level,rtol=0,atol=1e-14) and
            np.allclose(np.abs(truth.imag),level,rtol=0,atol=1e-14)):
        raise Invalid('only normalized QPSK symbols supported')
    # Tie at zero is deterministically the nonnegative bit; no oracle rescaling.
    def bits(x): return np.stack((x.real >= 0, x.imag >= 0),axis=-1)
    target_bits = bits(truth)
    a_errors = np.count_nonzero(bits(value) != target_bits,axis=(1,2,3))
    b_errors = np.count_nonzero(bits(reference) != target_bits,axis=(1,2,3))
    bits_per_channel = 2*truth.shape[1]*truth.shape[2]
    def evm(x):
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            return np.sqrt(np.sum(np.abs(x-truth)**2,axis=(1,2))/np.sum(np.abs(truth)**2,axis=(1,2)))
    a_evm,b_evm = evm(value),evm(reference)
    return [dict(batch=i,bits=bits_per_channel,candidate_bit_errors=int(a_errors[i]),
                 mmse_bit_errors=int(b_errors[i]),candidate_ber=float(a_errors[i]/bits_per_channel),
                 mmse_ber=float(b_errors[i]/bits_per_channel),ber_increase=float((a_errors[i]-b_errors[i])/bits_per_channel),
                 candidate_raw_symbol_evm=float(a_evm[i]),mmse_raw_symbol_evm=float(b_evm[i]),
                 evm_increase=float(a_evm[i]-b_evm[i])) for i in range(truth.shape[0])]


def summarize(groups, contract):
    """Simultaneous bounds across both metrics, all groups, and both tails."""
    if not groups: raise Invalid('no communication groups')
    alpha = contract['family_alpha'] / (4*len(groups))
    output = []
    for group in groups:
        rows=group['channels']; n=len(rows)
        if not n: raise Invalid('empty channel group')
        bounds={}
        for metric,limit in [('ber',contract['max_channel_ber_increase']),('evm',contract['max_channel_evm_increase'])]:
            k=sum(row[metric+'_increase']>limit for row in rows)
            bounds[metric]=outage_interval(k,n,alpha)
        upper_pass=all(b['upper']<=contract['max_outage_probability'] for b in bounds.values())
        lower_fail=any(b['lower']>contract['max_outage_probability'] for b in bounds.values())
        status=('QUALITY_FAILED' if lower_fail else 'QPSK_QUALITY_CHECKED'
                if upper_pass and n>=contract['min_independent_channels'] else 'INCONCLUSIVE')
        total_bits=sum(r['bits'] for r in rows)
        output.append({k:v for k,v in group.items() if k!='channels'} | dict(
            status=status,independent_channels=n,bits=total_bits,
            candidate_ber=sum(r['candidate_bit_errors'] for r in rows)/total_bits,
            mmse_ber=sum(r['mmse_bit_errors'] for r in rows)/total_bits,
            mean_candidate_raw_symbol_evm=float(np.mean([r['candidate_raw_symbol_evm'] for r in rows])),
            mean_mmse_raw_symbol_evm=float(np.mean([r['mmse_raw_symbol_evm'] for r in rows])),
            outage_bounds=bounds,tail_alpha=alpha,channels=rows))
    state=('QUALITY_FAILED' if any(g['status']=='QUALITY_FAILED' for g in output) else
           'QPSK_QUALITY_CHECKED' if all(g['status']=='QPSK_QUALITY_CHECKED' for g in output) else 'INCONCLUSIVE')
    return dict(status=state,groups=output,family_alpha=contract['family_alpha'],
                method='Clopper-Pearson per-channel outages; Bonferroni over groups/metrics/tails')
