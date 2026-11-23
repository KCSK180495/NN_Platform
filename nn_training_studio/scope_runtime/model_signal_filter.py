"""Declared model DSP, applied before scaling; independent of display settings."""
import numpy as np
from scipy import signal

FILTER_METHOD = 'Butterworth Low-pass'
FILTER_ID = 'LP500_BUTTER4_CAUSAL_20KHZ_V1'
DEFAULT_FILTER = dict(implementation='butterworth_sos_causal_v1',sample_rate_hz=20000.,
    cutoff_hz=500.,order=4,phase='causal',initial_state='first_sample_steady_state',warmup_samples=200)


def validate_signal_filter(spec, sample_rate):
    if not isinstance(spec,dict):
        raise ValueError('Model is missing its signal_filter definition')
    if spec.get('implementation')!='butterworth_sos_causal_v1' or spec.get('phase')!='causal':
        raise ValueError('Unsupported model signal filter implementation/phase')
    if spec.get('initial_state')!='first_sample_steady_state':
        raise ValueError('Unsupported model filter initial state')
    fs=float(spec.get('sample_rate_hz',0));cutoff=float(spec.get('cutoff_hz',0))
    order=spec.get('order',0);warmup=spec.get('warmup_samples',-1)
    if not np.isfinite([fs,cutoff,sample_rate]).all() or fs<=0 or not np.isclose(fs,sample_rate,rtol=1e-8,atol=0):
        raise ValueError('Model filter sample rate differs from input sample rate')
    if not 0<cutoff<fs/2 or isinstance(order,bool) or int(order)!=order or not 1<=order<=10:
        raise ValueError('Invalid model low-pass cutoff/order')
    if isinstance(warmup,bool) or int(warmup)!=warmup or not 0<=warmup<=10000:
        raise ValueError('Invalid model filter warmup')
    return dict(implementation='butterworth_sos_causal_v1',sample_rate_hz=fs,cutoff_hz=cutoff,
        order=int(order),phase='causal',initial_state='first_sample_steady_state',warmup_samples=int(warmup))


class SignalFilterStream:
    """Retain IIR state across chunks from one continuous acquisition only."""
    def __init__(self,spec,sample_rate):
        self.spec=validate_signal_filter(spec,sample_rate)
        self.sos=signal.butter(self.spec['order'],self.spec['cutoff_hz'],fs=sample_rate,output='sos')
        self.state=None
        self.samples=0

    def process(self,values):
        arr=np.asarray(values,dtype=float)
        if arr.ndim!=2 or not np.isfinite(arr).all():
            raise ValueError('Model filter needs finite samples by channels')
        if not len(arr):return arr.copy()
        if self.state is None:
            self.state=signal.sosfilt_zi(self.sos)[:,:,None]*arr[0][None,None,:]
        if self.state.shape[-1]!=arr.shape[1]:raise ValueError('Model filter channel count changed')
        output,self.state=signal.sosfilt(self.sos,arr,axis=0,zi=self.state)
        if not np.isfinite(output).all():raise ValueError('Model filter produced non-finite output')
        self.samples+=len(arr)
        return output


def apply_model_signal_filter(values,sample_rate,spec):
    return SignalFilterStream(spec,sample_rate).process(values)
