"""Canonical encoder implementation, isolated experiment-owned endpoint."""
import argparse,sys
from pathlib import Path
P=Path('/path/to/phyre-unc');S=P/'results/q0_first_feedback_20261018';R=P/'results/loss_design_iteration_20261019'
sys.path[:0]=[str(S/'sunflower_backend'),str(S/'code'),str(P/'pipeline')]
import encoder_rpc
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--lifetime',type=int,default=21600);a=p.parse_args()
encoder_rpc.B=R/f'encoder{a.index}'
# Keep encoder_rpc.R on the original frozen plan, and its code hash unchanged.
encoder_rpc.serve('0.0.0.0',0,'152.2.134.113',a.lifetime)
