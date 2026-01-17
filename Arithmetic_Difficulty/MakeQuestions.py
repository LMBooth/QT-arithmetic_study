"""Generate and pickle arithmetic question sets for the experiment."""
import QCalculator as qc
import pickle
questions = []

intmax = 30
for i in range(1,8):
    qmin = (i*0.9)-0.3
    qmax = ((i+1)*0.9)-0.3
    nums = qc.Generate_Q_Question(qmin, qmax, intmax)
    qs = [[qc.Generate_Q_Question(qmin, qmax, intmax) for i in range(10)], [qmin, qmax]]
    questions.append(qs)
    intmax = int(intmax * 2.5)


#print(qc.Generate_Q_Question(0.5, 1, 20))

with open('GeneratedQuestions', 'wb') as fp:
    pickle.dump(questions, fp)
    fp.close()

with open ('GeneratedQuestions', 'rb') as fp:
    questionsread = pickle.load(fp)
    fp.close()
    
print(questionsread) 
