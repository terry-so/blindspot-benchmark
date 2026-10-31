import ast
import base64
from io import BytesIO
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

from PIL import Image
import judge_results


class ClaudeJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'models/judges.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('OpenAIJudge', 'ClaudeJudge')]
        ns = dict(os=os, urlsplit=urlsplit, re=re, Image=Image, BytesIO=BytesIO, base64=base64,
                  Path=Path, __file__=str(Path(__file__).resolve().parents[1] / 'models/judges.py'))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'judges', 'exec'), ns)
        self.cls = ns['ClaudeJudge']
        self.sdk = Mock()
        self.sdk.return_value.messages.create.return_value = SimpleNamespace(content=[
            SimpleNamespace(type='thinking', thinking='not a score'),
            SimpleNamespace(type='text', text='<OUTPUT>1</OUTPUT>')])

    def test_foundry_route_key_image_order_and_score(self):
        with patch.dict('sys.modules', {'anthropic': SimpleNamespace(AnthropicFoundry=self.sdk)}), patch.dict(os.environ, {}, clear=True):
            judge = self.cls('claude-test', base_url='https://sample.openai.azure.com/openai/v1', api_key='test-key')
            self.sdk.assert_called_once_with(base_url='https://sample.services.ai.azure.com/anthropic',
                                             api_key='test-key', max_retries=0)
            original, edited = self.root / 'original.png', self.root / 'edited.png'
            Image.new('RGB', (8, 8), 'red').save(original)
            Image.new('RGB', (8, 8), 'blue').save(edited)
            self.assertEqual(judge.judge_fidelity(original, edited, 'target'), 1)
            request = self.sdk.return_value.messages.create.call_args.kwargs
            self.assertEqual(request['model'], 'claude-test')
            self.assertEqual(request['max_tokens'], 1024)
            content = request['messages'][0]['content']
            self.assertEqual(content[0], {'type': 'text', 'text': 'target'})
            self.assertEqual([b['type'] for b in content], ['text', 'image', 'image'])
            for block, expected in zip(content[1:], [(255, 0, 0), (0, 0, 255)]):
                self.assertEqual(block['source']['media_type'], 'image/png')
                with Image.open(BytesIO(base64.b64decode(block['source']['data']))) as image:
                    self.assertEqual(image.getpixel((0, 0)), expected)
            self.assertEqual(judge.detect_refusal('refusal prompt'), '<OUTPUT>1</OUTPUT>')
            self.assertEqual(len(self.sdk.return_value.messages.create.call_args.kwargs['messages'][0]['content']), 1)
            self.assertEqual(judge.judge_realism(edited), 1)

    def test_standard_routes_custom_override_and_error_propagation(self):
        with patch.dict('sys.modules', {'anthropic': SimpleNamespace(AnthropicFoundry=self.sdk)}), patch.dict(os.environ, {}, clear=True):
            for host in ['sample.services.ai.azure.com', 'sample.cognitiveservices.azure.com']:
                self.cls('claude-test', base_url=f'https://{host}/openai/v1/', api_key='key')
                self.assertEqual(self.sdk.call_args.kwargs['base_url'], 'https://sample.services.ai.azure.com/anthropic')
            with self.assertRaisesRegex(ValueError, 'custom endpoint'):
                self.cls('claude-test', base_url='https://custom.example/openai/v1', api_key='key')
            with patch.dict(os.environ, {'ANTHROPIC_FOUNDRY_BASE_URL': 'https://custom.example/claude'}):
                judge = self.cls('claude-test', base_url='https://custom.example/openai/v1', api_key='key')
                self.assertEqual(self.sdk.call_args.kwargs['base_url'], 'https://custom.example/claude')
            error = RuntimeError('unsupported route')
            self.sdk.return_value.messages.create.side_effect = error
            with self.assertRaisesRegex(RuntimeError, 'unsupported route'):
                judge.detect_refusal('test')

    def test_cli_selects_claude_only_for_azure_claude(self):
        for provider, model, expected in [('azure', 'claude-haiku-5-5-v2-156hgj', 'claude'),
                                           ('azure', 'gpt-test', 'openai'), ('openai', 'gpt-test', 'openai')]:
            openai, claude = Mock(), Mock()
            with patch.dict('sys.modules', {'models.judges': SimpleNamespace(OpenAIJudge=openai, ClaudeJudge=claude)}), \
                 patch.dict(os.environ, {'AZURE_OPENAI_BASE_URL': 'https://sample.services.ai.azure.com/openai/v1',
                                        'Azure_API_KEY': 'key', 'OPENAI_API_KEY': 'key'}), \
                 patch('evaluators.regrade.select_cases', return_value=([{}], {})), \
                 patch('evaluators.regrade.regrade', return_value={'failed': 0}) as grade:
                self.assertEqual(judge_results.main(['--run', str(self.root / 'source'), '--output', str(self.root / (provider + model)),
                                 '--metrics', 'fidelity', '--judge-provider', provider, '--judge-model', model]), 0)
                used, unused = (claude, openai) if expected == 'claude' else (openai, claude)
                used.assert_called_once()
                unused.assert_not_called()
                self.assertIs(grade.call_args.args[2], used.return_value)
