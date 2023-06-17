import pathlib

import jinja2
import openai


path = pathlib.Path("template.jinja2")

with path.open() as f:
    prompt_template = jinja2.Template(f.read())

labels = [
    {"label": 0, "description": "negative sentiment"},
    {"label": 1, "description": "neutral sentiment"},
    {"label": 2, "description": "positive sentiment"},
]

examples = [
    {"text": "Today was a horrible day", "label": 0},
    {"text": "Yesterday was a great day", "label": 2},
]

text = "I loved the TV show"

prompt = prompt_template.render(
    examples=examples,
    labels=labels,
    text=text,
)
print(prompt)

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message)
