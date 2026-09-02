from PyQt6.QtCore import QObject,pyqtSignal,pyqtSlot
from src.training.trainer import Trainer
class TrainingWorker(QObject):
    progress=pyqtSignal(int,str,str);completed=pyqtSignal(dict);failed=pyqtSignal(str);finished=pyqtSignal()
    def __init__(self,config):super().__init__();self.config=config
    @pyqtSlot()
    def run(self):
        try:self.completed.emit(Trainer(self.config,self.progress.emit).run())
        except Exception as exc:self.failed.emit(str(exc))
        finally:self.finished.emit()
