################ main.py ########################
# Written by Liam Booth 18/02/2023              #
#################################################
"""PyQt5 experiment runner that emits LSL markers during arithmetic trials."""
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
import pylsl
import pickle, random
import numpy as np

class MainExperiment(QtWidgets.QTabWidget):    
    """Tab widget that runs tutorial and experiment blocks."""
    currentQ = 0
    tutorial = True

    def __init__(self):        
        """Build the UI, load questions, and initialize the LSL stream."""
        super(MainExperiment, self).__init__()
        self.buttonBeginRoutine = QtWidgets.QPushButton('Begin Tutorial')
        self.buttonBeginRoutine.clicked.connect(self.BeginRoutine)
        self.buttonBeginRoutine.setStyleSheet("color: white")
        self.buttonBeginRoutine.setFont(QFont('Times', 36))
        self.text = QtWidgets.QLabel("In this experiment you will be presented a series of arithmetic \n "
                                     "questions. Each question will be presented for 6 seconds then an input \n "
                                     "box will appear. Keep still while calculating until this box appears.\n"
                                     "Input your answer with the keyboard and \n "
                                     "press space or enter to confirm. \n\n"
                                     "If you did not have time to calculate feel free to leave the box blank, \n"
                                     "again, Pressing space or enter to confirm. \n"
                                     "First you will be shown a tutorial with 8 random questions to \n"
                                     "familiarise yourself with the format. \n\n"
                                     "Please keep your head as still as possbile during the experiment\n"
                                     "and minimise your movements to the required keyboard presses.\n\n"
                                     
                                     "You will have a 1 minute blank period to start. \n "
                                     "Please keep your attention focused on x in the middle of the monitor for this minute.\n\n"
                                     
                                     
                                     "Please inform the experimenter if you are unsure of anything or \n"
                                     "press the begin tutorial text below to begin.")
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        self.text.setFont(QFont('Times', 36))
        self.text.setStyleSheet("color: white")
        self.setStyleSheet("background-color:black;")
        self.showMaximized()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text)
        self.lineEdit = QtWidgets.QLineEdit()
        self.lineEdit.returnPressed.connect(self.HandleAnswer)
        self.lineEdit.hide()
        self.lineEdit.setStyleSheet("QLineEdit { background-color: white; color: black }")
        font = self.lineEdit.font() 
        font.setPointSize(80)
        self.lineEdit.setFont(font) 
        self.lineEdit.installEventFilter(self)
        layout.addWidget(self.lineEdit)
        
        layout.addWidget(self.buttonBeginRoutine)
        with open ('GeneratedQuestions', 'rb') as fp:
            self.questions = pickle.load(fp)
            self.tutorialqs = [[q[0].pop(0), q[1]] for q in self.questions]
            random.shuffle(self.tutorialqs)
            random.shuffle(self.questions)
            fp.close()
        markerInfo = pylsl.StreamInfo("arithmetic-Markers", 'Markers', 1, 0, 'string', 'UoH')
        self.markerOutlet = pylsl.StreamOutlet(markerInfo)

    def BeginRoutine(self): 
        """Begin tutorial or experiment and schedule the first trial."""
        if self.tutorial:
            self.markerOutlet.push_sample(["Started tutorial artihmetic"])
        else:    
            self.markerOutlet.push_sample(["Started arithmetic"])
        self.buttonBeginRoutine.hide()
        self.text.setFont(QFont('Times', 80))
        if self.tutorial:
            self.text.setText("x")
            QTimer.singleShot(60000, self.DoRoutine)
        else:
            self.text.setText("")
            QTimer.singleShot(1000, self.DoRoutine)
    
    def DoRoutine(self):
        """Advance the trial state and present the next arithmetic prompt."""
        self.text.setFont(QFont('Times', 110))
        if self.tutorial: # run routine but base of tutorial trial counts, allows custom ending
            if len(self.tutorialqs) > 0:
                self.question = self.tutorialqs.pop(0)
                print(self.question)
                self.currentQ = self.question[1]
                s = str(self.question[0][0]) +" + "+ str(self.question[0][1])
                self.text.setText(s)
                QTimer.singleShot(6000, self.DoAnswerInputRoutine)
            else:
                self.tutorial = False
                self.text.setFont(QFont('Times', 50))
                self.text.setText("Tutorial Complete! \n Press the Begin Experiment text below when ready.")
                self.buttonBeginRoutine.show()
                self.buttonBeginRoutine.setText("Begin Experiment")
                self.markerOutlet.push_sample(["Finished tutorial Arithmetic"])
        else: # runs actual experiment count and ending
            i = sum(len(v[0]) for v in self.questions)
            if i > 0:
                rand = random.randint(0, len( self.questions)-1)
                self.question = [self.questions[rand][0].pop(0), self.questions[rand][1]]
                print(self.question)
                print(len(self.questions[rand][0]))
                if len(self.questions[rand][0]) <= 0:
                    self.questions.pop(rand)
                self.currentQ = self.question[1]
                s = str(self.question[0][0]) +" + "+ str(self.question[0][1])
                self.text.setText(s)
                QTimer.singleShot(6000, self.DoAnswerInputRoutine)
            else:   
                self.tutorial = True
                self.text.setFont(QFont('Times', 50))
                self.text.setText("Experiment finished! \n Please notify the Experimenter.")
                self.numberBack = 1
                self.markerOutlet.push_sample(["Finished Arithmetic"])

    def DoAnswerInputRoutine(self):
        """Show the answer input box and emit the difficulty marker."""
        self.text.hide()
        self.lineEdit.show()
        self.lineEdit.setFocus()
        s = str(round(self.currentQ[0],1)) +"-"+ str(round(self.currentQ[1],1))
        self.markerOutlet.push_sample([s])
    
    def eventFilter(self, source, event):
        """Handle key input in the answer box."""
        if (event.type() == QtCore.QEvent.KeyPress and source is self.lineEdit):
            if (event.key() == QtCore.Qt.Key_Enter) or  (event.key() == QtCore.Qt.Key_Space) or (event.key() == QtCore.Qt.Key_Return):
                self.HandleAnswer()
            elif (event.key() == QtCore.Qt.Key_Backspace):
                self.lineEdit.setText(self.lineEdit.text()[:-1])
            else:
                try:
                    self.lineEdit.setText(self.lineEdit.text()+event.text())
                    a = int(self.lineEdit.text()+event.text())
                except Exception as e:
                    self.lineEdit.setText(self.lineEdit.text()[:-1])
            return True
        else:    
            return super(MainExperiment, self).eventFilter(source, event)
    
    def HandleAnswer(self):
        """Record correctness marker and advance to the next trial."""
        self.text.show()
        self.text.setText("")
        self.lineEdit.hide()
        if self.lineEdit.text() == "":
            self.lineEdit.setText("0")
        if self.question[0][0] + self.question[0][1] == int(self.lineEdit.text()):
            s = str(round(self.currentQ[0],1)) +"-"+ str(round(self.currentQ[1],1)) + " Correct"
        else:
            s = str(round(self.currentQ[0],1)) +"-"+ str(round(self.currentQ[1],1)) + " Wrong"
        if self.tutorial:
            s = s +" tutorial"    
        print(s)    
        self.markerOutlet.push_sample([s])
        self.lineEdit.setText("")
        QTimer.singleShot(500, self.DoRoutine)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainExperiment()
    window.show()
    sys.exit(app.exec_())
