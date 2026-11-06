import io
import socket
import struct
import threading
import unittest
import zlib
import numpy as np
import encoder_rpc as rpc


class ProtocolTests(unittest.TestCase):
    def test_array_roundtrip(self):
        for a in [np.arange(128*256*256,dtype=np.uint8).reshape(128,256,256),
                  np.random.default_rng(41).normal(size=(32,196,128)).astype(np.float16)]:
            b=rpc.unpack(rpc.pack(a));np.testing.assert_array_equal(a,b);self.assertEqual(a.dtype,b.dtype)

    def test_socket_roundtrip_large_payload(self):
        key=b'local-test-key';a=np.random.default_rng(9).integers(0,255,(128,256,256),dtype=np.uint8)
        left,right=socket.socketpair()
        with left,right:
            t=threading.Thread(target=rpc.send,args=(left,{'nonce':'test'},rpc.pack(a),key));t.start()
            h,p=rpc.receive(right,key);t.join();self.assertEqual(h['nonce'],'test');np.testing.assert_array_equal(rpc.unpack(p),a)

    def test_reject_bad_auth(self):
        left,right=socket.socketpair()
        with left,right:
            rpc.send(left,{'nonce':'test'},b'hello',b'wrong')
            with self.assertRaisesRegex(ValueError,'authentication'):rpc.receive(right,b'right')

    def test_reject_truncated_message(self):
        left,right=socket.socketpair()
        left.sendall(b'bad');left.close()
        with right,self.assertRaises(ConnectionError):rpc.receive(right,b'key')

    def test_reject_oversize_header(self):
        left,right=socket.socketpair()
        with left,right:
            left.sendall(struct.pack('!IQ',65537,0))
            with self.assertRaisesRegex(ValueError,'Oversized'):rpc.receive(right,b'key')

    def test_reject_object_array_and_extra_data(self):
        f=io.BytesIO();np.save(f,np.array([{'bad':'object'}],object))
        with self.assertRaises(ValueError):rpc.unpack(zlib.compress(f.getvalue()))
        with self.assertRaises(ValueError):rpc.unpack(rpc.pack(np.arange(3))+b'extra')


if __name__=='__main__':unittest.main()
