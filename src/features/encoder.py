"""R3, encoder feature maps: a small 1D CNN with shared weights over sensors, and its training loop.

Architecture, applied to each sensor curve separately (shapes for one sensor, GWT / ES):
  input                                         (1, 1500)  / (1, 210)
  Conv1d(1 -> 16, k 7, stride 2, pad 3) + ReLU  (16, 750)  / (16, 105)
  Conv1d(16 -> 32, k 5, stride 2, pad 2) + ReLU (32, 375)  / (32, 53)
  global average pooling                        (32,)
All sensors flattened: F_enc (600, 2304) for GasWindTunnel, (155, 352) for EverydayScents.

Will provide:
- the encoder module (number of output channels configurable: 16 / 32 / 64 for C6);
- training inside a fold, on training data only: encoder + temporary linear layer on the class labels,
  with early stopping on an inner validation split; the linear layer is then removed and the encoder frozen;
- feature extraction with the frozen encoder;
- the end to end CNN classifier used as the reference point (03c).
"""
