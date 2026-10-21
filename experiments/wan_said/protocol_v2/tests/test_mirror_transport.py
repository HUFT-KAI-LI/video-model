import io
import unittest
from email.utils import formatdate
from unittest.mock import Mock, patch

import requests

from experiments.wan_said.protocol_v2.mirror_transport import (
    MIRROR, MirrorRangeSession, RequestGate, retry_after_seconds,
)


def response(status, headers=None):
    result = requests.Response()
    result.status_code = status
    result.url = MIRROR + '/test.zip'
    result.headers.update(headers or {})
    result.raw = io.BytesIO(b'abcd')
    return result


class MirrorTransportTests(unittest.TestCase):
    def test_retry_after_seconds_date_and_invalid(self):
        self.assertEqual(retry_after_seconds('125'), 125)
        self.assertEqual(retry_after_seconds(formatdate(1120, usegmt=True), now=1000), 120)
        self.assertEqual(retry_after_seconds('-10'), 0)
        self.assertIsNone(retry_after_seconds('invalid'))
        self.assertIsNone(retry_after_seconds(None))

    def test_429_honors_header_closes_response_and_retries_range(self):
        limited = response(429, {'Retry-After': '125'})
        success = response(206, {'Content-Range': 'bytes 0-3/4'})
        gate = Mock()
        with MirrorRangeSession(gate=gate) as session, \
             patch('requests.Session.request', side_effect=[limited, success]) as request:
            result = session.get(MIRROR + '/test.zip', headers={'Range': 'bytes=0-3'}, stream=True)
        self.assertIs(result, success)
        self.assertTrue(limited.raw.closed)
        gate.cooldown.assert_called_once_with(125)
        self.assertEqual(gate.wait.call_count, 2)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.kwargs['headers']['Range'], 'bytes=0-3')
        result.close()

    def test_429_without_header_backs_off_and_exhausts_without_other_host(self):
        gate = Mock()
        with MirrorRangeSession(gate=gate, rate_retries=2) as session, \
             patch('requests.Session.request', side_effect=lambda *a, **k: response(429)) as request:
            with self.assertRaises(requests.HTTPError):
                session.get(MIRROR + '/test.zip')
        self.assertEqual([call.args[0] for call in gate.cooldown.call_args_list], [60, 120, 240])
        self.assertTrue(all(call.args[1].startswith(MIRROR + '/') for call in request.call_args_list))

    def test_global_gate_shares_cooldown_and_paces_requests(self):
        gate = RequestGate(interval=.5)
        now = [100.0]
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        with patch('experiments.wan_said.protocol_v2.mirror_transport.time.monotonic', side_effect=lambda: now[0]), \
             patch('experiments.wan_said.protocol_v2.mirror_transport.time.sleep', side_effect=sleep):
            gate.wait()
            gate.wait()
            gate.cooldown(120)
            gate.wait()
        self.assertEqual(sleeps, [.5, 120])
        with MirrorRangeSession() as a, MirrorRangeSession() as b:
            self.assertIs(a.gate, b.gate)
            self.assertFalse(a.trust_env)

    def test_other_host_rejected_and_404_not_retried(self):
        gate = Mock()
        with MirrorRangeSession(gate=gate) as session, patch('requests.Session.request') as request:
            with self.assertRaises(ValueError):
                session.get('https://huggingface.co/test.zip')
            request.assert_not_called()
        missing = response(404)
        with MirrorRangeSession(gate=gate) as session, \
             patch('requests.Session.request', return_value=missing) as request:
            with self.assertRaises(requests.HTTPError):
                session.get(MIRROR + '/test.zip')
            request.assert_called_once()
        self.assertTrue(missing.raw.closed)
        gate.cooldown.assert_not_called()

    def test_server_ignoring_range_is_rejected(self):
        with MirrorRangeSession(gate=Mock()) as session, \
             patch('requests.Session.request', return_value=response(200)):
            with self.assertRaises(RuntimeError):
                session.get(MIRROR + '/test.zip', headers={'Range': 'bytes=0-3'})


if __name__ == '__main__':
    unittest.main()
