"""Explicit unit-energy Gray QPSK modem and full-link measurements."""
import math
import numpy as np
from verification.contracts.core import Invalid

LABELS=np.array([[1,1],[1,0],[0,1],[0,0]],dtype=np.uint8)
POINTS=((2*LABELS[:,0].astype(float)-1)+1j*(2*LABELS[:,1].astype(float)-1))/math.sqrt(2)
LABELS.flags.writeable=False;POINTS.flags.writeable=False


def modulate(bits):
    b=np.asarray(bits)
    if b.ndim<1 or b.shape[-1]!=2 or not b.size or b.dtype.kind not in 'biu' or not np.all((b==0)|(b==1)):
        raise Invalid('QPSK requires binary integer [...,2] I/Q labels')
    f=b.astype(np.float64)  # Do not subtract one in uint8 arithmetic.
    return ((2*f[...,0]-1)+1j*(2*f[...,1]-1))/math.sqrt(2)


def demodulate(symbols):
    z=np.asarray(symbols)
    if not z.size or not np.isfinite(z).all():raise Invalid('finite nonempty QPSK observations required')
    return np.stack((z.real>=0,z.imag>=0),axis=-1).astype(np.uint8)


def nearest(symbols):
    z=np.asarray(symbols)
    if not z.size or not np.isfinite(z).all():raise Invalid('finite nonempty distance observations required')
    with np.errstate(over='raise',invalid='raise'):
        distances=np.abs(z[...,None]-POINTS)**2
    # Label ordering resolves exact axis ties towards bit=1, as the sign slicer.
    return LABELS[np.argmin(distances,axis=-1)]


def theory(noise_variance):
    if not math.isfinite(noise_variance) or noise_variance<=0:raise Invalid('positive finite complex noise variance required')
    p=.5*math.erfc(1/math.sqrt(2*noise_variance))
    return dict(ber=p,ser=2*p-p*p,es_no_db=-10*math.log10(noise_variance),
                eb_no_db=-10*math.log10(2*noise_variance))


def bernoulli_tolerance(p,n,alpha):
    """Two-sided Bernstein bound with known Bernoulli mean (AWGN control only)."""
    if not 0<=p<=1 or type(n) is not int or n<=0 or not 0<alpha<1:raise Invalid('invalid Bernoulli control arguments')
    v=p*(1-p);L=math.log(2/alpha)
    return L/(3*n)+math.sqrt(2*v*L/n+(L/(3*n))**2)


def awgn_controls(seed=91001,symbol_count=200000):
    if type(seed) is not int or seed<0 or type(symbol_count) is not int or not 1000<=symbol_count<=2000000:
        raise Invalid('bounded AWGN control size/seed required')
    rows=[];levels=[-2.,0.,3.,6.,10.];alpha=.001/(2*len(levels))
    for index,db in enumerate(levels):
        rng=np.random.default_rng([seed,index]);bits=rng.integers(0,2,(symbol_count,2),dtype=np.uint8)
        x=modulate(bits);lam=10**(-db/10)
        n=math.sqrt(lam/2)*(rng.standard_normal(x.shape)+1j*rng.standard_normal(x.shape))
        soft=(x+n)/(1+lam);rx=demodulate(soft)
        if not np.array_equal(rx,nearest(soft)):raise Invalid('independent QPSK demappers disagree')
        bit_errors=int(np.count_nonzero(rx!=bits));symbol_errors=int(np.count_nonzero(np.any(rx!=bits,axis=-1)))
        ref=theory(lam);ber=bit_errors/(2*symbol_count);ser=symbol_errors/symbol_count
        bt=bernoulli_tolerance(ref['ber'],2*symbol_count,alpha);st=bernoulli_tolerance(ref['ser'],symbol_count,alpha)
        rows.append(dict(es_no_db=db,eb_no_db=ref['eb_no_db'],noise_variance=lam,bits=2*symbol_count,symbols=symbol_count,
                         bit_errors=bit_errors,symbol_errors=symbol_errors,ber=ber,ser=ser,theory_ber=ref['ber'],theory_ser=ref['ser'],
                         ber_tolerance=bt,ser_tolerance=st,passed=abs(ber-ref['ber'])<=bt and abs(ser-ref['ser'])<=st))
    allbits=LABELS.copy();ideal=demodulate(modulate(allbits)/(1+.1))
    return dict(schema='qpsk-control-v1',status='MODEM_CONTROLS_PASSED' if all(r['passed'] for r in rows) and np.array_equal(allbits,ideal) else 'MODEM_CONTROLS_FAILED',
                noiseless_identity_bit_errors=int(np.count_nonzero(allbits!=ideal)),awgn=rows,
                control_family_alpha=.001,method='known-mean Bernoulli Bernstein bound; IID AWGN bits/symbols only')


def measurements(candidate,mmse,bits):
    """Per-channel and per-stream counts; no oracle gain or phase fitting."""
    symbols=modulate(bits);a=np.asarray(candidate);b=np.asarray(mmse)
    if a.shape!=symbols.shape or b.shape!=symbols.shape or a.ndim!=3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise Invalid('matching [batch,Nt,RHS] link outputs required')
    ca=demodulate(a);cb=demodulate(b)
    if not np.array_equal(ca,nearest(a)) or not np.array_equal(cb,nearest(b)):
        raise Invalid('sign and independent minimum-distance demappers disagree')
    ba=ca!=bits;bb=cb!=bits;sa=np.any(ba,axis=-1);sb=np.any(bb,axis=-1)
    def counts(bit_bad,sym_bad,z):
        with np.errstate(over='raise',invalid='raise'):
            power=np.sum(np.abs(z-symbols)**2,axis=(1,2))
        return dict(bit_errors=np.count_nonzero(bit_bad,axis=(1,2,3)),symbol_errors=np.count_nonzero(sym_bad,axis=(1,2)),
                    stream_bit_errors=np.count_nonzero(bit_bad,axis=(2,3)),stream_symbol_errors=np.count_nonzero(sym_bad,axis=2),
                    squared_error=power)
    def confusion(rx):
        txid=(2*bits[...,0]+bits[...,1]).reshape(-1);rid=(2*rx[...,0]+rx[...,1]).reshape(-1)
        return np.bincount(4*txid+rid,minlength=16).reshape(4,4)
    return dict(candidate=counts(ba,sa,a),mmse=counts(bb,sb,b),
                candidate_confusion=confusion(ca),mmse_confusion=confusion(cb),
                symbols_per_channel=symbols.shape[1]*symbols.shape[2],bits_per_channel=2*symbols.shape[1]*symbols.shape[2])
