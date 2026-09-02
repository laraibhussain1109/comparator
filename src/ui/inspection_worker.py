from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import json,cv2,time
from PyQt6.QtCore import QThread,pyqtSignal
from src.inspection.pipeline import InspectionPipeline
from src.inspection.temporal_filter import TemporalConfirmation
from src.inspection.part_state_machine import PartStateMachine

class InspectionWorker(QThread):
    frame_ready=pyqtSignal(object,str,dict);error=pyqtSignal(str);event=pyqtSignal(str)
    def __init__(self,config):super().__init__();self.c=config;self.running=True
    def stop(self):self.running=False;self.requestInterruption()
    def _save(self,original,result):
        now=datetime.now(timezone.utc);directory=Path(self.c["paths"]["results"])/now.strftime("%Y-%m-%d");directory.mkdir(parents=True,exist_ok=True);stem=now.strftime("%H%M%S_%f")
        cv2.imwrite(str(directory/f"{stem}_original.jpg"),original);cv2.imwrite(str(directory/f"{stem}_marked.jpg"),result.marked_image);payload={"timestamp":now.isoformat(),"result":result.result,"model_scores":result.scores,"geometry":result.geometry,"registration_confidence":result.registration_confidence,"detected_regions":[r.to_dict() for r in result.regions]};(directory/f"{stem}_result.json").write_text(json.dumps(payload,indent=2))
    def run(self):
        cap=None
        try:
            pipeline=InspectionPipeline(self.c);temporal=TemporalConfirmation(self.c["temporal"]["window"],self.c["temporal"]["confirmation_count"]);state=PartStateMachine(self.c["inspection"]["absence_frames"])
            start=self.c["camera"]["index"]
            for index in [start]+[i for i in range(self.c["camera"]["probe_count"]) if i!=start]:
                candidate=cv2.VideoCapture(index)
                if candidate.isOpened():cap=candidate;self.event.emit(f"Camera {index} connected");break
                candidate.release()
            if cap is None:raise RuntimeError("No camera could be opened. Check the connection and camera index in Settings/config.yaml.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.c["camera"]["width"]);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.c["camera"]["height"])
            while self.running and not self.isInterruptionRequested():
                ok,frame=cap.read()
                if not ok:time.sleep(.05);continue
                result=pipeline.inspect(frame);confirmed=temporal.update(result.result,result.geometry.get("critical",False)) if result.part_present else "RECHECK";_,count=state.update(result.part_present,confirmed if confirmed!="RECHECK" else None)
                shown=confirmed if result.part_present else "RECHECK";details={"scores":result.scores,"registration":result.registration_confidence,"count_event":count}
                self.frame_ready.emit(result.marked_image,shown,details)
                if count:
                    self.event.emit(shown)
                    if (shown=="NG" and self.c["inspection"]["save_ng_events"]) or (shown=="GOOD" and self.c["inspection"]["save_good_events"]):self._save(frame,result)
                if not result.part_present:temporal.reset()
        except Exception as exc:self.error.emit(str(exc))
        finally:
            if cap is not None:cap.release()
