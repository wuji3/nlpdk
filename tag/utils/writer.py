import pandas as pd
import json
from typing import Union

class ShardByShardWriter:
    def __init__(self, ftype: str = 'csv') -> None:
        assert ftype in {'csv', 'jsonl'}, "Only support csv and jsonl"
        self.ftype = ftype
        self.first = True

    def write_shard2csv(self, shard: pd.DataFrame, filepath: str) -> None:
        assert isinstance(shard, pd.DataFrame), "Must be a pd.DataFrame."
        if self.first:
            shard.to_csv(filepath, mode='w', header=True, index=False)
            self.first = False
        else: shard.to_csv(filepath, mode='a', header=False, index=False)
    
    def write_shard2jsonl(self, shard: dict, filepath: str) -> None:
        assert isinstance(shard, dict), "Must be a dict."
        if self.first:
            with open(filepath, 'w') as f:
                f.write(json.dumps(shard) + '\n')
            self.first = False
        else:
            with open(filepath, 'a') as f:
                f.write(json.dumps(shard) + '\n')
    
    def __call__(self, shard: Union[pd.DataFrame, dict], filepath: str) -> None:
        assert isinstance(shard, (pd.DataFrame, dict)), "Must be a pd.DataFrame or a dict."
        if self.ftype == 'csv': self.write_shard2csv(shard, filepath)
        elif self.ftype == 'jsonl': self.write_shard2jsonl(shard, filepath)