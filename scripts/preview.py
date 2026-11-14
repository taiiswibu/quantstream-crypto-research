"""Adapt the managed preview port to Streamlit."""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', default='4173')
args, _ = parser.parse_known_args()
os.execv(sys.executable, [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.address', args.host, '--server.port', args.port, '--server.headless', 'true'])
