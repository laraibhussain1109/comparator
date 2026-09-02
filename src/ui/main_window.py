from __future__ import annotations
from pathlib import Path
import json,platform,time
import cv2
from PyQt6.QtCore import Qt,QThread,QTimer
from PyQt6.QtGui import QImage,QPixmap
from PyQt6.QtWidgets import *
from src.ui.training_worker import TrainingWorker
from src.ui.inspection_worker import InspectionWorker

STYLE="""QWidget{background:#111820;color:#e8edf2;font-family:Arial}QPushButton{background:#263747;border:1px solid #587086;border-radius:5px;padding:12px;font-weight:bold;font-size:15px}QPushButton:hover{background:#345067}QPushButton:disabled{color:#65717c;background:#202a32}QProgressBar{border:1px solid #4a5d6d;border-radius:4px;text-align:center}QProgressBar::chunk{background:#168bd2}QGroupBox{border:1px solid #344657;border-radius:5px;margin-top:8px;padding-top:12px}QTextEdit{background:#0b1117;border:1px solid #344657}"""
class MainWindow(QMainWindow):
    def __init__(self,config):
        super().__init__();self.c=config;self.counts={"Total":0,"GOOD":0,"NG":0,"RECHECK":0};self.inspector=None;self.training_thread=None;self.started=0;self.setWindowTitle("AI Casting Inspection System");self.setStyleSheet(STYLE);self._build();self._fit_to_screen();self._refresh_trained()
    def _fit_to_screen(self):
        screen=self.screen()
        available=screen.availableGeometry() if screen else None
        self.resize(min(1250,available.width()) if available else 1250,min(900,available.height()) if available else 900)
    def _build(self):
        root=QWidget();layout=QVBoxLayout(root);title=QLabel("AI CASTING INSPECTION SYSTEM");title.setAlignment(Qt.AlignmentFlag.AlignCenter);title.setStyleSheet("font-size:27px;font-weight:bold;padding:12px;color:#dceaf5");layout.addWidget(title)
        self.video=QLabel("LIVE CAMERA IMAGE\n\nPlace a component in view after training");self.video.setAlignment(Qt.AlignmentFlag.AlignCenter);self.video.setMinimumHeight(180);self.video.setStyleSheet("background:#05090c;border:2px solid #344657;font-size:18px;color:#789");layout.addWidget(self.video,1)
        self.result=QLabel("RESULT: READY");self.result.setAlignment(Qt.AlignmentFlag.AlignCenter);self.result.setStyleSheet("font-size:36px;font-weight:bold;color:#8fa6b8;padding:12px");layout.addWidget(self.result)
        stats=QHBoxLayout();self.stat_labels={}
        for key in self.counts:
            box=QGroupBox(key if key!="Total" else "Total Inspected");v=QVBoxLayout(box);label=QLabel("0");label.setAlignment(Qt.AlignmentFlag.AlignCenter);label.setStyleSheet("font-size:29px;font-weight:bold");v.addWidget(label);stats.addWidget(box);self.stat_labels[key]=label
        layout.addLayout(stats);buttons=QHBoxLayout();self.train=QPushButton("TRAIN");self.start=QPushButton("START INSPECTION");self.stop=QPushButton("STOP INSPECTION");self.diagnostics=QPushButton("RUN DIAGNOSTICS");self.stop.setEnabled(False);buttons.addWidget(self.train);buttons.addWidget(self.start);buttons.addWidget(self.stop);buttons.addWidget(self.diagnostics);layout.addLayout(buttons)
        self.progress=QProgressBar();self.progress.setVisible(False);layout.addWidget(self.progress);self.stage=QLabel("");layout.addWidget(self.stage);self.log=QTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(115);layout.addWidget(self.log)
        tech=QGroupBox("Technical Details");tech.setCheckable(True);tech.setChecked(False);tv=QVBoxLayout(tech);self.technical=QLabel(f"Hardware: {platform.platform()} | Python {platform.python_version()}\nCamera index: {self.c['camera']['index']} | Image size: {self.c['image_size']}");tv.addWidget(self.technical);tech.toggled.connect(lambda checked:self.technical.setVisible(checked));self.technical.setVisible(False);layout.addWidget(tech)
        self.setCentralWidget(root);self.train.clicked.connect(self.train_models);self.start.clicked.connect(self.start_inspection);self.stop.clicked.connect(self.stop_inspection);self.diagnostics.clicked.connect(lambda:self.train_models(True))
    def _refresh_trained(self):self.start.setEnabled((Path(self.c["paths"]["artifacts"])/"training_summary.json").exists())
    def train_models(self,diagnostics=False):
        self.train.setEnabled(False);self.start.setEnabled(False);self.progress.setVisible(True);self.progress.setValue(0);self.started=time.monotonic();self.training_thread=QThread();self.worker=TrainingWorker(self.c,diagnostics);self.worker.moveToThread(self.training_thread);self.training_thread.started.connect(self.worker.run);self.worker.progress.connect(self.on_progress);self.worker.completed.connect(self.on_training_done);self.worker.failed.connect(self.on_error);self.worker.finished.connect(self.training_thread.quit);self.worker.finished.connect(lambda:self.train.setEnabled(True));self.training_thread.start()
    def on_progress(self,p,stage,message):self.progress.setValue(p);self.stage.setText(f"Current Stage: {stage}   •   Elapsed: {time.monotonic()-self.started:.0f}s");self.log.append(message)
    def on_training_done(self,summary):
        if "test" not in summary:self.result.setText("SANITY CHECK PASSED");self.log.append(f"Sanity checkpoint reload: {summary['sanity_check']['correct']} / {summary['sanity_check']['total']}");return
        t=summary["test"];self.result.setText("TRAINING COMPLETED");self.result.setStyleSheet("font-size:32px;font-weight:bold;color:#35c66b;padding:12px");self.log.append(f"Held-out test: accuracy {t['accuracy']:.1%}, NG recall {t['ng_recall']:.1%}, precision {t['precision']:.1%}, F2 {t['f2']:.1%}");self._refresh_trained()
    def on_error(self,message):QMessageBox.critical(self,"Inspection system",message);self.log.append("ERROR: "+message);self.train.setEnabled(True);self._refresh_trained()
    def start_inspection(self):
        if self.inspector and self.inspector.isRunning():return
        try:self.inspector=InspectionWorker(self.c);self.inspector.frame_ready.connect(self.on_frame);self.inspector.error.connect(self.on_error);self.inspector.event.connect(self.on_event);self.inspector.finished.connect(lambda:self.stop.setEnabled(False));self.inspector.start();self.start.setEnabled(False);self.stop.setEnabled(True);self.train.setEnabled(False)
        except Exception as exc:self.on_error(str(exc))
    def stop_inspection(self):
        if self.inspector:self.inspector.stop();self.inspector.wait(3000)
        self.stop.setEnabled(False);self.train.setEnabled(True);self._refresh_trained();self.result.setText("RESULT: STOPPED")
    def on_frame(self,image,result,details):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);h,w,c=rgb.shape;q=QImage(rgb.data,w,h,c*w,QImage.Format.Format_RGB888).copy();self.video.setPixmap(QPixmap.fromImage(q).scaled(self.video.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation));colors={"GOOD":"#35c66b","NG":"#ff3b3b","RECHECK":"#ffc247"};self.result.setText("RESULT: "+result);self.result.setStyleSheet(f"font-size:36px;font-weight:bold;color:{colors[result]};padding:12px");self.technical.setText(f"Registration: {details['registration']:.3f} | Internal model scores: {json.dumps(details['scores'])}")
    def on_event(self,event):
        if event in self.counts and event!="Total":self.counts["Total"]+=1;self.counts[event]+=1
        else:self.log.append(event)
        for key,label in self.stat_labels.items():label.setText(str(self.counts[key]))
    def closeEvent(self,event):self.stop_inspection();event.accept()
