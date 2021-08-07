import sys
import os
sys.stdout = sys.stderr
activate_this = '/home/jose/.virtualenvs/covfeenew/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))
os.chdir('/data/lared-annotation')
from covfee.server.start import create_app
application = create_app('deploy')[1]