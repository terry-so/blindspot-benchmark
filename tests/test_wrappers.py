"""Exercise the actual Azure wrapper class with a fake SDK, no cloud imports."""
import ast
import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
import os
import tempfile
from unittest.mock import Mock, patch
from PIL import Image

class WrapperTests(unittest.TestCase):
    def setUp(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'models/wrappers.py').read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'_OpenAIImageBase', 'AzureImageEditor'}]
        class APIStatusError(Exception):
            def __init__(self,code,body=None):self.status_code=code;self.body=body
        self.error=APIStatusError
        namespace={'APIStatusError':APIStatusError,'Image':Image,'BytesIO':BytesIO,'base64':base64,
                   'record_usage':lambda response:None}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'wrapper','exec'),namespace)
        self.base=namespace['_OpenAIImageBase']
        self.model=namespace['AzureImageEditor'].__new__(namespace['AzureImageEditor'])
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


class OpenAIImageTests(WrapperTests):
    def setUp(self):
        super().setUp()
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'models/wrappers.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OpenAIImageEditor')
        self.sdk = Mock()
        namespace = {'_OpenAIImageBase': self.base, 'OpenAI': self.sdk, 'os': os}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'wrapper', 'exec'), namespace)
        self.wrapper = namespace['OpenAIImageEditor']
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True):
            self.model = self.wrapper('test-model')

    def test_direct_credentials_and_endpoint(self):
        self.sdk.assert_called_once_with(base_url='https://api.openai.com/v1',
                                         api_key='test-key', max_retries=0)

    def test_missing_key_fails_before_request(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(KeyError):
            self.wrapper('test-model')

    def test_edit_saves_image_and_closes_input(self):
        buffer = BytesIO()
        Image.new('RGB', (8, 8), 'red').save(buffer, format='PNG')
        self.model.client.images.edit.return_value = SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(buffer.getvalue()).decode())],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {
                'input_tokens_details': {'text_tokens': 10, 'image_tokens': 20},
                'output_tokens': 30,
            }))
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'output.png'
            result = self.model.generate_edit('change color', __file__, target)
            self.assertEqual(result['status'], 'success')
            self.assertFalse(result['safety_block'])
            self.assertEqual(result['usage']['output_tokens'], 30)
            from utils.benchmark_runner import response_metadata
            self.assertEqual(response_metadata(result)['usage'], result['usage'])
            with Image.open(target) as saved:
                self.assertEqual(saved.size, (8, 8))
            args = self.model.client.images.edit.call_args.kwargs
            self.assertEqual(args['model'], 'test-model')
            self.assertEqual(args['prompt'], 'change color')
            self.assertTrue(args['image'].closed)

    def test_empty_image_is_error_not_refusal(self):
        self.model.client.images.edit.return_value = SimpleNamespace(data=[])
        with self.assertRaises(RuntimeError):
            self.model.generate_edit('edit', __file__, 'unused.png')
