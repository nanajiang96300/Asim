"""Sionna CDL-B static frequency snapshots, not the legacy Kronecker surrogate."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from verification.contracts.core import Invalid

SPEC=dict(model='B',carrier_frequency_hz=3.5e9,delay_spread_s=100e-9,
          frequency_offset_hz=1.2e6,direction='uplink',ut_rows=4,ut_cols=4,
          bs_rows=8,bs_cols=8,polarization='single',polarization_type='V',
          antenna_pattern='omni',spacing_wavelengths=.5,speed_m_s=0.,
          time_steps=1,sampling_frequency_hz=30000.,precision='double',device='cpu',
          normalization='per_snapshot_frobenius_squared_Nt')


def provenance():
    import sionna
    import torch
    base=Path(sionna.__file__).parent
    paths=sorted([*base.rglob('*.py'),*base.rglob('*.json')])
    return dict(sionna=importlib.metadata.version('sionna'),torch=torch.__version__,
                numpy=np.__version__,source_files={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def normalize(H):
    H=np.asarray(H,dtype=np.complex128)
    if H.ndim!=3 or not np.isfinite(H).all():raise Invalid('invalid channel matrix')
    power=np.sum(np.abs(H)**2,axis=(1,2))
    if np.any(power<=0):raise Invalid('zero channel energy')
    return H*np.sqrt(H.shape[2]/power)[:,None,None]


def frequency_response(a,tau,frequency_hz):
    a=np.asarray(a);tau=np.asarray(tau)
    if a.ndim!=4 or tau.shape!=(a.shape[0],a.shape[-1]) or not np.isfinite(a).all() or not np.isfinite(tau).all():
        raise Invalid('CDL path shape/finite mismatch')
    return np.sum(a*np.exp(-2j*np.pi*frequency_hz*tau)[:,None,None,:],axis=-1)


def generate(seed,batch=96,spec=None):
    spec=SPEC if spec is None else spec
    if spec!=SPEC:raise Invalid('unsupported CDL-B profile; freeze a new explicit version')
    if type(seed) is not int or seed<0 or type(batch) is not int or not 1<=batch<=192:raise Invalid('CDL seed/batch budget')
    import torch
    from sionna.phy import config
    from sionna.phy.channel.tr38901 import CDL,PanelArray
    from sionna.phy.channel import cir_to_ofdm_channel
    config.seed=seed
    kw=dict(polarization='single',polarization_type='V',antenna_pattern='omni',
            carrier_frequency=spec['carrier_frequency_hz'],precision='double',device='cpu',
            element_vertical_spacing=.5,element_horizontal_spacing=.5)
    ut=PanelArray(4,4,**kw);bs=PanelArray(8,8,**kw)
    cdl=CDL('B',spec['delay_spread_s'],spec['carrier_frequency_hz'],ut,bs,'uplink',
            min_speed=0.,max_speed=0.,precision='double',device='cpu')
    values=[];coeff=[];delays=[];disagreement=[]
    # Fixed one-link chunks avoid a multi-GB ray/antenna tensor on the 2-vCPU host.
    with torch.no_grad():
        for _ in range(batch):
            a,t=cdl(1,1,spec['sampling_frequency_hz'])
            aa=a[:,0,:,0,:,:,0].cpu().numpy();tt=t[:,0,0,:].cpu().numpy()
            direct=frequency_response(aa,tt,spec['frequency_offset_hz'])
            reference=cir_to_ofdm_channel(torch.tensor([spec['frequency_offset_hz']],dtype=torch.float64),a,t,normalize=False)
            ref=reference[:,0,:,0,:,0,0].cpu().numpy()
            diff=float(np.max(np.abs(direct-ref)));disagreement.append(diff)
            if not np.allclose(direct,ref,rtol=1e-12,atol=1e-12):raise Invalid('independent CIR frequency conversion mismatch')
            values.append(direct[0]);coeff.append(aa[0]);delays.append(tt[0])
    return dict(H=normalize(np.stack(values)),path_coefficients=np.stack(coeff),path_delays_s=np.stack(delays),
                conversion_max_abs_error=np.array(max(disagreement)))
