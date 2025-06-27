"""Implementation of the Bonito-CTC model

Based on: 
https://github.com/nanoporetech/bonito
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from classes import BaseModelImpl
from layers.bonito import BonitoLSTM
from s5 import S5Block


class BonitoModel(BaseModelImpl):
    """Bonito Model
    """
    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True, load_default = False,
                 nlstm=0,slstm_threshold=0.05, conv_threshold=0, *args, **kwargs):
        super(BonitoModel, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            reverse (bool): if the first rnn layer starts with reverse 
        """
    
        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder
        self.reverse = reverse
        
        if load_default:
            self.load_default_configuration()

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch, channels (1), len]
        """
        
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def build_cnn(self):


        cnn = nn.Sequential(
            nn.Conv1d(          #4, 4, 2000
                in_channels = 1, 
                out_channels = 4, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(         #4, 16, 2000
                in_channels = 4, 
                out_channels = 16, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(           #4, 384, 400
                in_channels = 16, 
                out_channels = 384, 
                kernel_size = 19, 
                stride= 5, 
                padding=19//2, 
                bias=True),
            nn.SiLU()
        )
        return cnn
    

    def build_encoder(self, input_size, reverse):

        if reverse:
            encoder = nn.Sequential(BonitoLSTM(input_size, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True))
        else:
            encoder = nn.Sequential(BonitoLSTM(input_size, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False))   ########## BonitoSLSTM : MODIFICA SPIKING! ##########
        return encoder    

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 384, 
            'cnn_output_activation': 'silu',
            'encoder_input_size': 384,
            'encoder_output_size': 384,
            'cnn_stride': 5,
        }
        return defaults
        
    def load_default_configuration(self):
        """Sets the default configuration for one or more
        modules of the network
        """

        self.convolution = self.build_cnn()
        self.cnn_stride = self.get_defaults()['cnn_stride']
        self.encoder = self.build_encoder(input_size = 384, reverse = True)
        self.decoder = self.build_decoder(encoder_output_size = 384, decoder_type = 'crf')
        self.decoder_type = 'crf'

class S5Model(BaseModelImpl):
    """S5 Model
    """
    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True, load_default = False,
                 nlstm=0,slstm_threshold=0.05, conv_threshold=0, *args, **kwargs):
        super(S5Model, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            reverse (bool): if the first rnn layer starts with reverse 
        """
    
        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder
        self.reverse = reverse
        self.nblock = nlstm
        
        if load_default:
            self.load_default_configuration()

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch, channels (1), len]
        """
        
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def build_cnn_version_0(self):

        cnn = nn.Sequential(
            nn.Conv1d(          #4, 4, 2000
                in_channels = 1, 
                out_channels = 16, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(         #4, 16, 2000
                in_channels = 16, 
                out_channels = 64, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(           #4, 384, 400
                in_channels = 64, 
                out_channels = 384, 
                kernel_size = 19, 
                stride= 5, 
                padding=19//2, 
                bias=True),
            nn.SiLU()
        )
        return cnn
    
    def build_cnn_version_1(self):
        # Try this CNN
        cnn = nn.Sequential(
            # Layer 1: Individual base signals
            nn.Conv1d(          # batch, 64, 2000
                in_channels = 1, 
                out_channels = 64, 
                kernel_size = 5, 
                stride = 1, 
                padding = 5//2, 
                bias = True),
            nn.LayerNorm([64, 2000]),
            nn.SiLU(),
            nn.Dropout1d(0.05),
            
            # Layer 2: Base transitions (dilation=2)
            nn.Conv1d(          # batch, 96, 2000
                in_channels = 64, 
                out_channels = 96, 
                kernel_size = 5, 
                stride = 1, 
                padding = 4,     # For dilation=2
                dilation = 2,
                bias = True),
            nn.LayerNorm([96, 2000]),
            nn.SiLU(),
            nn.Dropout1d(0.05),
            
            # Layer 3: K-mer context (dilation=4)
            nn.Conv1d(          # batch, 128, 2000
                in_channels = 96, 
                out_channels = 128, 
                kernel_size = 5, 
                stride = 1, 
                padding = 8,     # For dilation=4
                dilation = 4,
                bias = True),
            nn.LayerNorm([128, 2000]),
            nn.SiLU(),
            nn.Dropout1d(0.1),
            
            # Layer 4: Homopolymer runs (dilation=8 + stride=2)
            nn.Conv1d(          # batch, 256, 1000
                in_channels = 128, 
                out_channels = 256, 
                kernel_size = 7, 
                stride = 2, 
                padding = 24,    # Adjusted for dilation=8
                dilation = 8,
                bias = True),
            nn.LayerNorm([256, 1000]),
            nn.SiLU(),
            nn.Dropout1d(0.1),
            
            # Layer 5: Final integration
            nn.Conv1d(          # batch, 512, 400
                in_channels = 256, 
                out_channels = 512, 
                kernel_size = 9, 
                stride = 2, 
                padding = 9//2, 
                bias = True),
            nn.LayerNorm([512, 500]),
            nn.SiLU(),
            nn.Dropout1d(0.15),
            
            nn.AdaptiveAvgPool1d(400) # with and without it no conseguences
        )
        return cnn

    def build_cnn_version_2(self):
        # Light CNN
        cnn = nn.Sequential(
            # Stem: Initial signal processing
            nn.Conv1d(1, 24, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Dropout1d(0.1),
            
            # Stage 1: Individual base signal detection - Block 1
            nn.Conv1d(24, 72, kernel_size=1, bias=False),  # Expand (24 * 3)
            nn.GroupNorm(8, 72),
            nn.SiLU(),
            nn.Conv1d(72, 72, kernel_size=5, stride=1, padding=2, groups=72, bias=False),  # Depthwise
            nn.GroupNorm(8, 72),
            nn.SiLU(),
            nn.Conv1d(72, 32, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(4, 32),
            nn.Dropout1d(0.05),
            
            # Stage 1: Individual base signal detection - Block 2 (with residual-like structure)
            nn.Conv1d(32, 64, kernel_size=1, bias=False),  # Expand (32 * 2)
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, groups=64, bias=False),  # Depthwise
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 32, kernel_size=1, bias=False),  # Compress (same output as input for residual)
            nn.GroupNorm(4, 32),
            nn.Dropout1d(0.05),
            nn.SiLU(),
            
            # Stage 2: Base transition detection - Block 1
            nn.Conv1d(32, 96, kernel_size=1, bias=False),  # Expand (32 * 3)
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv1d(96, 96, kernel_size=7, stride=1, padding=6, dilation=2, groups=96, bias=False),  # Depthwise
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv1d(96, 48, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(6, 48),
            nn.Dropout1d(0.05),
            
            # Stage 2: Base transition detection - Block 2
            nn.Conv1d(48, 96, kernel_size=1, bias=False),  # Expand (48 * 2)
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=1, padding=4, dilation=2, groups=96, bias=False),  # Depthwise
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv1d(96, 48, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(6, 48),
            nn.Dropout1d(0.05),
            nn.SiLU(),
            
            # Stage 3: K-mer context modeling - Block 1 (with downsampling)
            nn.Conv1d(48, 192, kernel_size=1, bias=False),  # Expand (48 * 4)
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            nn.Conv1d(192, 192, kernel_size=5, stride=2, padding=8, dilation=4, groups=192, bias=False),  # Depthwise + downsample
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            nn.Conv1d(192, 64, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(8, 64),
            nn.Dropout1d(0.05),
            
            # Stage 3: K-mer context modeling - Block 2
            nn.Conv1d(64, 128, kernel_size=1, bias=False),  # Expand (64 * 2)
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=4, dilation=2, groups=128, bias=False),  # Depthwise
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, 64, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(8, 64),
            nn.Dropout1d(0.05),
            nn.SiLU(),
            
            # Stage 4: Homopolymer run detection - Block 1 (with downsampling)
            nn.Conv1d(64, 256, kernel_size=1, bias=False),  # Expand (64 * 4)
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=12, dilation=4, groups=256, bias=False),  # Depthwise + downsample
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 96, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(8, 96),
            nn.Dropout1d(0.05),
            
            # Stage 4: Homopolymer run detection - Block 2
            nn.Conv1d(96, 192, kernel_size=1, bias=False),  # Expand (96 * 2)
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            nn.Conv1d(192, 192, kernel_size=5, stride=1, padding=4, dilation=2, groups=192, bias=False),  # Depthwise
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            nn.Conv1d(192, 96, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(8, 96),
            nn.Dropout1d(0.05),
            nn.SiLU(),
            
            # Stage 5: Final integration
            nn.Conv1d(96, 288, kernel_size=1, bias=False),  # Expand (96 * 3)
            nn.GroupNorm(8, 288),
            nn.SiLU(),
            nn.Conv1d(288, 288, kernel_size=5, stride=1, padding=2, groups=288, bias=False),  # Depthwise
            nn.GroupNorm(8, 288),
            nn.SiLU(),
            nn.Conv1d(288, 128, kernel_size=1, bias=False),  # Compress to final output
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            
            # Final adaptive pooling to exact output length
            nn.AdaptiveAvgPool1d(400)   # [batch, 128, 400]
        )
        return cnn
    
    def build_cnn(self):
        cnn = nn.Sequential(
            # Stem: Initial signal processing
            nn.Conv1d(1, 24, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Dropout1d(0.1),
            
            # Stage 1: Individual base signal detection - Block 1
            nn.Conv1d(24, 72, kernel_size=1, bias=False),  # Expand (24 * 3)
            nn.GroupNorm(8, 72),
            nn.SiLU(),
            nn.Conv1d(72, 72, kernel_size=5, stride=1, padding=2, groups=72, bias=False),  # Depthwise
            nn.GroupNorm(8, 72),
            nn.SiLU(),
            nn.Conv1d(72, 32, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(4, 32),
            nn.Dropout1d(0.05),
            
            # Stage 1: Individual base signal detection - Block 2 (with residual-like structure)
            nn.Conv1d(32, 64, kernel_size=1, bias=False),  # Expand (32 * 2)
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, groups=64, bias=False),  # Depthwise
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 32, kernel_size=1, bias=False),  # Compress (same output as input for residual)
            nn.GroupNorm(4, 32),
            nn.Dropout1d(0.05),
            nn.SiLU(),
            
            # Stage 2: Base transition detection - Block 1
            nn.Conv1d(32, 96, kernel_size=1, bias=False),  # Expand (32 * 3)
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv1d(96, 96, kernel_size=7, stride=1, padding=6, dilation=2, groups=96, bias=False),  # Depthwise
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv1d(96, 48, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(6, 48),
            nn.Dropout1d(0.05),
            
            # Stage 2: Base transition detection - Block 2
            nn.Conv1d(48, 96, kernel_size=1, bias=False),  # Expand (48 * 2)
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=1, padding=4, dilation=2, groups=96, bias=False),  # Depthwise
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv1d(96, 48, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(6, 48),
            nn.Dropout1d(0.05),
            nn.SiLU(),
            
            # Stage 3: K-mer context modeling - Block 1 (with downsampling)
            nn.Conv1d(48, 192, kernel_size=1, bias=False),  # Expand (48 * 4)
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            nn.Conv1d(192, 192, kernel_size=5, stride=2, padding=8, dilation=4, groups=192, bias=False),  # Depthwise + downsample
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            nn.Conv1d(192, 64, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(8, 64),
            nn.Dropout1d(0.05),
            
            # Stage 3: K-mer context modeling - Block 2
            nn.Conv1d(64, 128, kernel_size=1, bias=False),  # Expand (64 * 2)
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=4, dilation=2, groups=128, bias=False),  # Depthwise
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, 64, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(8, 64),
            nn.Dropout1d(0.05),
            nn.SiLU(),
            
            # Stage 4: Homopolymer run detection - Block 1 (with downsampling)
            nn.Conv1d(64, 256, kernel_size=1, bias=False),  # Expand (64 * 4)
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=12, dilation=4, groups=256, bias=False),  # Depthwise + downsample
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 128, kernel_size=1, bias=False),  # Compress to 128
            nn.GroupNorm(8, 128),
            nn.Dropout1d(0.05),
            
            # Stage 4: Homopolymer run detection - Block 2
            nn.Conv1d(128, 256, kernel_size=1, bias=False),  # Expand (128 * 2)
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=4, dilation=2, groups=256, bias=False),  # Depthwise
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 128, kernel_size=1, bias=False),  # Compress
            nn.GroupNorm(8, 128),
            nn.Dropout1d(0.05),
            nn.SiLU(),
            
            # Stage 5: Final integration - increase to 256 channels
            nn.Conv1d(128, 384, kernel_size=1, bias=False),  # Expand (128 * 3)
            nn.GroupNorm(8, 384),
            nn.SiLU(),
            nn.Conv1d(384, 384, kernel_size=5, stride=1, padding=2, groups=384, bias=False),  # Depthwise
            nn.GroupNorm(8, 384),
            nn.SiLU(),
            nn.Conv1d(384, 256, kernel_size=1, bias=False),  # Compress to 256 channels (CHANGED)
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            
            # Final adaptive pooling to exact output length
            nn.AdaptiveAvgPool1d(400)
        )
        return cnn

    def build_encoder(self, input_size, reverse):
        
        encoder = S5Block(
            dim=input_size,
            state_dim=96,   # even with 246 dim no big changes
            bidir=reverse,
            block_count=self.nblock,
            ff_dropout=0.0,
        )
        return encoder    

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 384, 
            'cnn_output_activation': 'silu',
            'encoder_input_size': 384,
            'encoder_output_size': 384,
            'cnn_stride': 5,
        }
        return defaults
        
    def load_default_configuration(self):
        """Sets the default configuration for one or more
        modules of the network
        """

        self.convolution = self.build_cnn()
        self.cnn_stride = self.get_defaults()['cnn_stride']
        self.encoder = self.build_encoder(input_size = 128, reverse = True)
        self.decoder = self.build_decoder(encoder_output_size = 128, decoder_type = 'crf')
        self.decoder_type = 'crf'

