"""Serve an installed test consumer and a local model, without copying the model."""
import argparse
import functools
import http.server
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('consumer')
parser.add_argument('model')
parser.add_argument('--port', type=int, default=8520)
args = parser.parse_args()
model = Path(args.model)

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ('/model.gguf', '/slow-model.gguf'):
            return super().do_GET()
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(model.stat().st_size))
        self.end_headers()
        try:
            with model.open('rb') as stream:
                while chunk := stream.read(65536):
                    self.wfile.write(chunk)
                    if self.path == '/slow-model.gguf':
                        time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass

http.server.ThreadingHTTPServer(('127.0.0.1', args.port), functools.partial(Handler, directory=args.consumer)).serve_forever()
