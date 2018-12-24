from mxnet import gluon
from mxnet.gluon import nn, HybridBlock
import mxnet as mx

class PAM_Module(HybridBlock):
    """ Position attention module
        From paper: Dual Attention Network for Scene Segmentation
    """
    def __init__(self, in_dim, **kwargs):
        super(PAM_Module, self).__init__(**kwargs)
        self._channel_in = in_dim
        self.query_conv = nn.Conv2D(channels=in_dim // 8, kernel_size=1, in_channels=in_dim)
        self.key_conv = nn.Conv2D(channels=in_dim // 8, kernel_size=1, in_channels=in_dim)
        self.value_conv = nn.Conv2D(in_channels=in_dim, channels=in_dim, kernel_size=1)
        self.gamma = self.params.get('gamma', shape=(1,),
                                          init=mx.init.Zero(),
                                          allow_deferred_init=True)
    
    def hybrid_forward(self, F, x, gamma):
        proj_query = self.query_conv(x).reshape((0, self._channel_in // 8, -1))
        proj_key = self.key_conv(x).reshape((0, self._channel_in // 8, -1))
        energy = F.batch_dot(proj_query, proj_key, transpose_a=True)
        attention = F.softmax(energy)
        proj_value = self.value_conv(x).reshape((0, self._channel_in, -1))
        out = F.batch_dot(proj_value, attention, transpose_b=True)
        out = F.broadcast_mul(gamma, F.reshape_like(out, x)) + x
        return out

class PAM_ModuleWithoutGamma(HybridBlock):
    """ Position attention module
        From paper: Dual Attention Network for Scene Segmentation
    """
    def __init__(self, in_dim, **kwargs):
        super(PAM_ModuleWithoutGamma, self).__init__(**kwargs)
        self._channel_in = in_dim
        self.query_conv = nn.Conv2D(channels=in_dim // 8, kernel_size=1, in_channels=in_dim)
        self.key_conv = nn.Conv2D(channels=in_dim // 8, kernel_size=1, in_channels=in_dim)
        self.value_conv = nn.Conv2D(in_channels=in_dim, channels=in_dim, kernel_size=1)
    
    def hybrid_forward(self, F, x):
        proj_query = self.query_conv(x).reshape((0, self._channel_in // 8, -1))
        proj_key = self.key_conv(x).reshape((0, self._channel_in // 8, -1))
        energy = F.batch_dot(proj_query, proj_key, transpose_a=True)
        attention = F.softmax(energy)
        proj_value = self.value_conv(x).reshape((0, self._channel_in, -1))
        out = F.batch_dot(proj_value, attention, transpose_b=True)
        out = F.reshape_like(out, x) + x
        return out

class CAM_Module(HybridBlock):
    """ Channel attention module
        From paper: Dual Attention Network for Scene Segmentation
    """
    def __init__(self, in_dim, **kwargs):
        super(CAM_Module, self).__init__(**kwargs)
        self._channel_in = in_dim
        self.gamma = self.params.get('gamma', shape=(1,),
                                          init=mx.init.Zero(),
                                          allow_deferred_init=True)

    def hybrid_forward(self, F, x, gamma):
        reshaped_x = x.reshape((0, self._channel_in, -1))
        energy = F.batch_dot(reshaped_x, reshaped_x, transpose_b=True)
        attention = F.softmax(-(F.max(energy, -1, keepdims=True).broadcast_like(energy) - energy))
        proj_value = x.reshape((0, self._channel_in, -1))
        out = F.batch_dot(attention, proj_value)
        out = F.broadcast_mul(gamma, F.reshape_like(out, x)) + x
        return out
        #from IPython import embed; embed()

if __name__ == '__main__':
    pam = PAM_Module(in_dim=16)
    pam.initialize()
    pam.hybridize()
    #input_ = mx.ndarray.ones((2, 16, 32, 32))
    input_ = mx.nd.random.uniform(0, 1, shape=(2, 16, 32, 32))
    out_ = pam(input_)
    cam = CAM_Module(in_dim=16)
    cam.initialize()
    cam.hybridize()
    out_ = cam(input_)
    from IPython import embed; embed()