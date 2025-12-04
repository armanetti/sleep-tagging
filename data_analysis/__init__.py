import os
import os.path as op
import sys
repo = op.abspath('.')

if repo not in os.sys.path:
    sys.path.append(repo)