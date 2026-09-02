from __future__ import annotations
from enum import Enum,auto

class PartState(Enum):NO_PART=auto();PART_ENTERED=auto();INSPECTING=auto();RESULT_LOCKED=auto();PART_EXITED=auto()
class PartStateMachine:
    def __init__(self,absence_frames=4):self.state=PartState.NO_PART;self.absence_frames=absence_frames;self.absent=0
    def update(self,present:bool,confirmed_result:str|None=None)->tuple[PartState,bool]:
        count=False
        if present:
            self.absent=0
            if self.state in (PartState.NO_PART,PartState.PART_EXITED):self.state=PartState.PART_ENTERED
            elif self.state==PartState.PART_ENTERED:self.state=PartState.INSPECTING
            if confirmed_result in ("GOOD","NG","RECHECK") and self.state==PartState.INSPECTING:self.state=PartState.RESULT_LOCKED;count=True
        else:
            self.absent+=1
            if self.absent>=self.absence_frames:
                if self.state==PartState.RESULT_LOCKED:self.state=PartState.PART_EXITED
                elif self.state==PartState.PART_EXITED:self.state=PartState.NO_PART
                elif self.state!=PartState.NO_PART:self.state=PartState.NO_PART
        return self.state,count
