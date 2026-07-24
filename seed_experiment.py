import os

import autoevals
from autoevals import Factuality
from braintrust import Eval
from openai import AsyncOpenAI, OpenAI

FIREWORKS_MODEL = "accounts/fireworks/models/gpt-oss-120b"
FIREWORKS_KWARGS = {
    "api_key": os.environ["FIREWORKS_API_KEY"],
    "base_url": "https://api.fireworks.ai/inference/v1",
}

client = OpenAI(**FIREWORKS_KWARGS)

# Grade with Fireworks too, since this Braintrust org has no OpenAI
# provider configured for the AI gateway that Factuality defaults to.
# Braintrust always invokes scorers through autoevals' async path, so
# the judge client must be an AsyncOpenAI client.
autoevals.init(client=AsyncOpenAI(**FIREWORKS_KWARGS), default_model=FIREWORKS_MODEL)

DATASET = [
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is the capital of Japan?", "expected": "Tokyo"},
    {"input": "Who wrote Romeo and Juliet?", "expected": "William Shakespeare"},
    {"input": "What is the chemical symbol for gold?", "expected": "Au"},
    {"input": "What is the largest planet in our solar system?", "expected": "Jupiter"},
    {"input": "In what year did World War II end?", "expected": "1945"},
    {"input": "What is the tallest mountain in the world?", "expected": "Mount Everest"},
    {"input": "Who painted the Mona Lisa?", "expected": "Leonardo da Vinci"},
    {"input": "What is the smallest prime number?", "expected": "2"},
    {"input": "What is the currency of Japan?", "expected": "Yen"},
    {"input": "How many continents are there on Earth?", "expected": "7"},
    {"input": "What is the freezing point of water in Celsius?", "expected": "0"},
    {"input": "Who was the first president of the United States?", "expected": "George Washington"},
    {"input": "What is the largest ocean on Earth?", "expected": "Pacific Ocean"},
    {"input": "What gas do plants absorb from the atmosphere for photosynthesis?", "expected": "Carbon dioxide"},
    {"input": "What is the capital of Italy?", "expected": "Rome"},
    {"input": "Who developed the theory of general relativity?", "expected": "Albert Einstein"},
    {"input": "What is the longest river in the world?", "expected": "Nile River"},
    {"input": "What is the boiling point of water in Celsius?", "expected": "100"},
    {"input": "How many sides does a hexagon have?", "expected": "6"},
]

BASELINE_SYSTEM_PROMPT = "Answer the question."
CANDIDATE_SYSTEM_PROMPT = "Answer the question accurately, in as few words as possible."
CANDIDATE_MARGINAL_SYSTEM_PROMPT = "Answer the question. Be accurate and precise."


def _ask(system_prompt, question):
    response = client.chat.completions.create(
        model=FIREWORKS_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def baseline_task(input):
    return _ask(BASELINE_SYSTEM_PROMPT, input)


def candidate_task(input):
    return _ask(CANDIDATE_SYSTEM_PROMPT, input)


def candidate_marginal_task(input):
    return _ask(CANDIDATE_MARGINAL_SYSTEM_PROMPT, input)


PROJECT_NAME = "eval-reliability-demo"

Eval(
    PROJECT_NAME,
    experiment_name="baseline",
    data=DATASET,
    task=baseline_task,
    scores=[Factuality],
)

Eval(
    PROJECT_NAME,
    experiment_name="candidate",
    data=DATASET,
    task=candidate_task,
    scores=[Factuality],
)

Eval(
    PROJECT_NAME,
    experiment_name="candidate_marginal",
    data=DATASET,
    task=candidate_marginal_task,
    scores=[Factuality],
)
