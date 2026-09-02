from PyQt6.QtCore import QObject,pyqtSignal,pyqtSlot
from src.training.trainer import Trainer
class TrainingWorker(QObject):
    progress=pyqtSignal(int,str,str);completed=pyqtSignal(dict);failed=pyqtSignal(str);finished=pyqtSignal()
    def __init__(self,config,diagnostics=False):super().__init__();self.config=config;self.diagnostics=diagnostics
    @pyqtSlot()
    def run(self):
        try:
            trainer=Trainer(self.config,self.progress.emit);self.completed.emit(trainer.run_diagnostics() if self.diagnostics else trainer.run())
        except Exception as exc:self.failed.emit(str(exc))
        finally:self.finished.emit()
