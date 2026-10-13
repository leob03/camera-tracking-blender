"""Log protocol/client tests without Blender, ComfyUI, models or GPU work."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vipe_remote.client import Client, Detached
from vipe_remote.job_logs import LogBuffer, LogFollower
spec = importlib.util.spec_from_file_location('log_tail', ROOT/'comfyui_vipe_remote/log_tail.py')
tail = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tail)
ID = '00000000-0000-4000-8000-000000000001'


class LocalClient:
    def __init__(self, path):
        self.path = path

    def log_chunk(self, identifier, offset, generation):
        return tail.read_log_chunk(self.path, offset, generation)


class JobLogs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'worker.log'

    def test_missing_append_unicode_no_duplicates_and_truncation(self):
        output=[]
        follower=LogFollower(LocalClient(self.path), ID, output.append)
        follower.poll()
        self.assertEqual(output, [])
        self.path.write_bytes(b'a'*32767 + '😊'.encode()[:1])
        follower.offset=0
        follower.poll()
        with self.path.open('ab') as f: f.write('😊'.encode()[1:]+b'\n')
        follower.poll();follower.poll()
        self.assertEqual(''.join(output), 'a'*32767+'😊\n')
        self.path.write_text('restarted\n')
        follower.poll()
        self.assertIn('Worker log restarted', ''.join(output))
        self.assertTrue(''.join(output).endswith('restarted\n'))

    def test_rotation_and_bounded_backlog(self):
        self.path.write_bytes(b'x'*(tail.MAX_BACKLOG_BYTES*2))
        chunk=tail.read_log_chunk(self.path, 0)
        self.assertEqual(chunk['skipped_bytes'], tail.MAX_BACKLOG_BYTES)
        self.assertEqual(chunk['next_offset']-chunk['offset'],tail.CHUNK_BYTES)
        replacement=self.path.with_suffix('.new');replacement.write_text('new log\n');replacement.replace(self.path)
        reset=tail.read_log_chunk(self.path,0,chunk['generation'])
        self.assertTrue(reset['reset'])
        with self.assertRaises(ValueError):tail.read_log_chunk(self.path,-2)

    def test_final_failure_output_and_nonfatal_logging(self):
        path=self.path;path.write_text('starting\n');output=[]
        class FailedJob(Client):
            def log_chunk(self, identifier, offset, generation):
                return tail.read_log_chunk(path,offset,generation)
            def request(self, method, endpoint, data=None):
                if endpoint.startswith('/history/'):
                    with path.open('a') as f:f.write('last traceback line\n')
                    return {ID:{'status':{'completed':False,'status_str':'error','messages':[['execution_error',{'exception_message':'solve failed'}]]}}}
                raise AssertionError(endpoint)
        with self.assertRaisesRegex(RuntimeError,'solve failed'):
            FailedJob('http://localhost').wait(ID,log=output.append)
        self.assertIn('last traceback line', ''.join(output))
        class CompletedJob(FailedJob):
            def log_chunk(self,*args):raise OSError('temporary log outage')
            def request(self,*args):return {ID:{'outputs':{'1':{'vipe_result':[{'job_id':ID}]}}}}
        self.assertEqual(CompletedJob('http://localhost').wait(ID,log=output.append),{'job_id':ID})
        # Display callback failures likewise do not replace the job result.
        def broken_display(text):raise RuntimeError('closed terminal')
        self.assertEqual(CompletedJob('http://localhost').wait(ID,log=broken_display),{'job_id':ID})

    def test_detach_does_not_cancel_worker(self):
        class DetachedClient(Client):
            def cancel(self, identifier):raise AssertionError('detach must not cancel')
            def log_chunk(self,*args):return dict(exists=False)
        with self.assertRaises(Detached):DetachedClient('http://localhost').wait(ID,detach=lambda:True,log=lambda text:None)

    def test_real_http_cursor_closes_connections(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from urllib.parse import urlsplit, parse_qs
        path=self.path
        path.write_text('HTTP first line\n')
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                query=parse_qs(urlsplit(self.path).query)
                body=json.dumps(tail.read_log_chunk(path,int(query['offset'][0]),query.get('generation',[None])[0])).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)))
                self.end_headers();self.wfile.write(body)
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            client=Client(f'http://127.0.0.1:{server.server_port}')
            output=[];follower=LogFollower(client,ID,output.append)
            follower.poll()
            with path.open('a') as stream:stream.write('HTTP second line\n')
            follower.poll()
            self.assertEqual(''.join(output),'HTTP first line\nHTTP second line\n')
            for _ in range(20):follower.poll()
            self.assertEqual(len(output),2)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)

    def test_display_buffer_bounds_and_carriage_returns(self):
        buffer=LogBuffer(limit=32)
        buffer.append('\x1b[31mred\x1b[0m\r');buffer.append('\nnext\rupdate\n')
        self.assertEqual(buffer.snapshot()[1], 'red\nnext\nupdate\n')
        buffer.append('x'*1000)
        self.assertLessEqual(len(buffer.text),32)
        self.assertTrue(buffer.trimmed)
        self.assertIn('full log remains on the worker',buffer.snapshot()[1])


if __name__=='__main__':unittest.main()
