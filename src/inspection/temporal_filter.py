from __future__ import annotations
from collections import deque,Counter


class TemporalConfirmation:
    def __init__(self,window=5,confirm=3):self.values=deque(maxlen=window);self.confirm=confirm
    def update(self,value:str,critical=False)->str:
        if value=="RECHECK":return value
        self.values.append(value)
        if critical and value=="NG":return "NG"
        counts=Counter(self.values)
        if counts["NG"]>=self.confirm:return "NG"
        if len(self.values)>=self.confirm and counts["GOOD"]>=self.confirm:return "GOOD"
        return "RECHECK"
    def reset(self)->None:self.values.clear()
