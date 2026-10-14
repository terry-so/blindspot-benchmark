"""Exercise the actual Azure wrapper class with a fake SDK, no cloud imports."""
import ast
import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from PIL import Image

class WrapperTests(unittest.TestCase):
    def setUp(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'models/wrappers.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Azure_gpt_image')
        class APIStatusError(Exception):
            def __init__(self,code,body=None):self.status_code=code;self.body=body
        self.error=APIStatusError
        namespace={'APIStatusError':APIStatusError,'Image':Image,'BytesIO':BytesIO,'base64':base64,
                   'record_usage':lambda response:None}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'wrapper','exec'),namespace)
        self.model=namespace['Azure_gpt_image'].__new__(namespace['Azure_gpt_image'])
        self.model.model_name='mock'
    def call_error(self,error):
        def edit(**kwargs):raise error
        self.model.client=SimpleNamespace(images=SimpleNamespace(edit=edit))
        return self.model.generate_edit('edit',__file__,'unused.png')
    def test_transient_status_propagates_to_retry_layer(self):
        with self.assertRaises(self.error):self.call_error(self.error(429))
    def test_invalid_argument_propagates_without_safety_label(self):
        with self.assertRaises(self.error):self.call_error(self.error(400,{'code':'INVALID_ARGUMENT'}))
    def test_provider_safety_is_preserved(self):
        response=self.call_error(self.error(400,{'error':{'innererror':{'code':'content_filter'}}}))
        self.assertTrue(response['safety_block'])
        self.assertEqual(response['block_reason'],'content_filter')
        self.assertEqual(response['status'],'failed')
