try:
    from .tgn_sequential import TGNSequential, TGNCarryOver, TGNLstm
except ImportError:
    from model.tgn_sequential import TGNSequential, TGNCarryOver, TGNLstm

try:
    from .tgn_depression import TGNDepression
except ImportError:
    TGNDepression = None
