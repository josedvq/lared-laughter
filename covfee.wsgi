import sys
import os
sys.stdout = sys.stderr
activate_this = '/home/jose/.virtualenvs/lared/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))
os.chdir('/home/jose/lared-annotation')
from covfee.start import create_app
application = create_app()