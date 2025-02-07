## Installing

- run install.sh (to install hardware dependent torch automatically)

### Manual Installing

For GPU:

`poetry install --extras "gpu"`

For CPU:

`poetry install --extras "cpu"`


## Data Generation

Prompt to manually generate ground truth from Qwen 2.5 VL 72b (from the chat UI):

"""
Please extract all the informations that you find on the document in a json format.
Note: The json keys must be in english (snake case) and the values in the document's original language.
"""

## TODO:

- in poetry add all dependencies related to training under [train] extra
- make all empty null or ""
- add a mechanism to artificially add null elements on documents to train the model to not answer when element is not present (tricky)
- wait for llama cpu to release support to run qwen2 vl 2b