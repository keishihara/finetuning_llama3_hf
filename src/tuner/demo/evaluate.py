import json
import warnings
from pathlib import Path

import numpy as np
import requests
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import AutoTokenizer
from tuner.demo.predict import PredictionWriter
from typer import Typer

warnings.simplefilter('ignore')
app = Typer()

RETRY_COUNT = 3
SYSTEM_PROMPT = 'You are a Japanese text evaluator who always answers in Japanese.'
# adapted from: https://soysoftware.sakura.ne.jp/archives/3850
PROMPT_TEMPLATE = """
以下には問題、正解例、採点基準、被験者の回答が与えられます。
最後の指示に従って採点結果を答えてください。

# 問題
{input}

# 正解例
{output}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{eval_aspect}

# 被験者の回答
{prediction}

// ここまでが「被験者回答」です。回答が空白だった場合、1点にしてください。

# 指示
「採点基準」および「正解例」を参考にして、「問題」に対する「被験者の回答」を1~5の5段階で採点し数字のみを出力してください。
"""

# Llama3 model is assumed as the evaluator
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

with Path.open(Path(__file__).parent / 'elyza_tasks_100.gbnf') as f:
    grammar = f.read()


def format_messages(prediction: str, input: str, output: str, eval_aspect: str) -> str:
    prompt = PROMPT_TEMPLATE.format(
        input=input.strip(),
        output=output.strip(),
        eval_aspect=eval_aspect.strip(),
        prediction=prediction.strip(),
    )
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]


@retry(wait=wait_exponential(min=5, max=30), stop=stop_after_attempt(RETRY_COUNT))
def compute(
    prediction: str,
    reference: dict[str, str],
    url: str = 'http://127.0.0.1:8080/completion',
) -> int:
    """Computes the evaluation score for a given prediction.

    Args:
        prediction (str): The predicted text.
        reference (dict[str, str]): The reference data containing 'input',
            'output', and 'eval_aspect'.
        url (str, optional): The URL of the evaluation server. Defaults to
            'http://127.0.0.1:8080/completion'.

    Returns:
        int: The evaluation score.

    Raises:
        RuntimeError: If the server response is invalid.
        ValueError: If the generated score is not valid.
    """
    if prediction == '': # when the prediction is an empty char
        return 1
    inp = reference['input']
    out = reference['output']
    eval_aspect = reference['eval_aspect']
    messages = format_messages(prediction, inp, out, eval_aspect)
    eval_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    data = {
        'prompt': eval_prompt,
        'n_predict': 1, # limit the num chars to generate to 1.
        'temperature': 0.6,
        'top_k': 10, # this might be too much
        'top_p': 0.5,
        'cache_prompt': True,
        'grammar': grammar,
    }
    data = json.dumps(data)
    r = requests.post(
        url,
        data=data,
        headers={'Content-Type': 'application/json'},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f'Failed to get a valid response from server: {r.status_code} - {r.content}')
    d = json.loads(r.content)
    try:
        score = int(d['content'])
    except ValueError as e:
        raise ValueError('Not a valid score generated: ' + d['content']) from e
    if score not in (1,2,3,4,5):
        raise ValueError('Score is not in valid range: ' + d['content'])
    return score


class ResultsWriter(PredictionWriter):
    """
    A class to handle writing scores to a file.

    Args:
        file_path (str): The path to the output file.
    """
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.file = None
        self.content = None
        self.dump_kwargs = {'indent': 4, 'ensure_ascii': False}

        if not Path(file_path).is_file():
            raise FileNotFoundError(f'Provided `file_path` not found: {file_path}')

    def __enter__(self) -> 'ResultsWriter':
        self = super().__enter__()
        if 'scores' not in self.content:
            self.content['scores'] = {}
        return self

    def log(self, index: int | str, scores: list[int]) -> None:
        """
        Args:
            index (int | str): The index of the prediction.
            score (list[int]): The elyza-tasks-100 score
        """
        if str(index) in self.content['scores']:
            raise ValueError(f'A score already exist at `{index = }`.')
        self.content['scores'][str(index)] = scores
        self._save_to_file()

    def _save_to_file(self) -> None:
        messages = self.content['messages']
        scores = self.content['scores']
        if scores and len(scores) == len(messages):
            self.content['ELYZA-tasks-100'] = np.mean(list(scores.values()), axis=1).mean()
        super()._save_to_file()


@app.command()
def evaluate(
    inp_file: str,
    *,
    n_repeat: int = 3,
    silent: bool = False,
    url: str = 'http://127.0.0.1:8080/completion',
) -> None:
    """Evaluates the predictions against the ELYZA-tasks-100 dataset.

    This function assumes that an evaluator LLM is running locally using the
    HTTP server feature of llama.cpp. Ensure that the server is launched before
    running this function. Also, predictions on the ELYZA-tasks-100 by the
    evaluatee model must be generated in advance (online evaluation is not supported).

    Args:
        inp_file (str): The input file containing predictions.
        n_repeat (int, optional): The number of times to repeat evaluation for
            each prediction. Defaults to 3.
        silent (bool, optional): If True, disables progress output. Defaults to
            False.
        url (str, optional): The URL of the evaluation server. Defaults to
            'http://127.0.0.1:8080/completion'.

    Raises:
        FileExistsError: If the result file already exists and overwrite is False.
        ValueError: If the last item in prediction is not an assistant message.

    References:
        elyza/ELYZA-tasks-100: https://huggingface.co/datasets/elyza/ELYZA-tasks-100
    """
    inp_file = Path(inp_file)
    # load test dataset
    refs = load_dataset('elyza/ELYZA-tasks-100', split='test')
    # load predicted assistat messages
    with Path.open(inp_file) as f:
        data = json.load(f)
    preds = [data['messages'][str(i)][-1] for i in range(len(refs))]

    with ResultsWriter(inp_file) as writer:
        all_scores = []
        for i, (pred, ref) in tqdm(
            enumerate(zip(preds, refs, strict=True)),
            desc='Computing scores..',
            dynamic_ncols=True,
            total=len(preds),
            disable=silent,
        ):
            if pred['role'] != 'assistant':
                raise ValueError(f'The provided message is not an assistant message: {pred}')
            scores = [compute(pred['content'], ref, url=url) for _ in range(n_repeat)]
            writer.log(i, scores)
            all_scores.append(scores)

    if not silent:
        all_scores = np.array(all_scores)
        print(f'Final score: {all_scores.mean(axis=1).mean()}')
        print(f'Results are saved at: {inp_file.as_posix()}')


if __name__ == '__main__':
    app()
