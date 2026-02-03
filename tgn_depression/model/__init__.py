try:
    from .tgn_sequential import TGNSequential, TGNCarryOver, TGNLstm
except ImportError:
    from model.tgn_sequential import TGNSequential, TGNCarryOver, TGNLstm
