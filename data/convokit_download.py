# from convokit import download
import pandas as pd

# from sqlalchemy.engine import create_engine
# from sqlalchemy.engine.url import URL


DATA_DIR = "/data/avirinchipur/EMI/raw_data/CONVO_AWRY"

# AWRY_ROOT_DIR = download('conversations-gone-awry-corpus', data_dir=DATA_DIR)

utterances = pd.read_json(DATA_DIR + '/conversations-gone-awry-corpus' + "/utterances.jsonl", lines=True)
conversations = pd.read_json(DATA_DIR + '/conversations-gone-awry-corpus' + "/conversations.json").T

import pdb; pdb.set_trace()