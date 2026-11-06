from common import *
import socket
statuses=[]
for i in range(2):
 dest=R/f'encoder{i}/rpc';(dest/'STOP').touch()
 # Wake the listening socket after STOP. No authenticated data or credentials sent.
 endpoint=read(dest/'ENDPOINT.json')
 try:
  with socket.create_connection((endpoint['host'],endpoint['port']),timeout=5):pass
 except OSError:pass
for attempt in range(30):
 statuses=[read(R/f'encoder{i}/rpc/STATUS.json') for i in range(2)]
 if all(s.get('stage')=='stopped' for s in statuses):break
 time.sleep(2)
else:raise RuntimeError('Owned encoder stop not confirmed')
write(R/'ENCODER_CLEANUP.json',{'complete':True,'statuses':statuses,'time':time.time()});print('ENCODERS_STOPPED')
