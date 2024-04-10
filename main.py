import pandas
import numpy
from matplotlib import pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

penguins = pandas.read_csv("palmer_penguins.csv")
penguins.head()

penguins = penguins.drop(["studyName", "Sample Number", "Region", "Stage", "Individual ID", "Clutch Completion",
            "Date Egg", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Comments"], axis = 1)
penguins["Species"] = penguins["Species"].str.split().str.get(0)
penguins = penguins[penguins["Sex"] != "."]
penguins = penguins.dropna(subset = ["Sex"])
penguins.head()
#print(penguins)

def prep_penguins_data(data):
    dframe = data.copy()
    encoder = preprocessing.LabelEncoder()

    dframe["Sex"] = encoder.fit_transform(dframe["Sex"])
    dframe["Island"] = encoder.fit_transform(dframe["Island"])
    dframe["Species"] = encoder.fit_transform(dframe["Species"])

    Pred = dframe.drop(["Species"], axis = 1)
    Target = dframe["Species"]

    return Pred, Target

train, test = train_test_split(penguins, test_size=0.4)
Pred_Train, Targer_Train = prep_penguins_data(train)

def column_scores(column):

    LR = LogisticRegression(max_iter=5000)

    return cross_val_score(LR, Pred_Train[column], Targer_Train, cv = 5).mean()

quals = ["Island", "Sex"]

quants = ["Culmen Length (mm)", "Culmen Depth (mm)",
        "Flipper Length (mm)", "Body Mass (g)"]

combos = [[qual]+[quant1]+[quant2] for qual in quals for quant1 in quants for quant2 in quants if quant1 != quant2]

cv_scores = []
best_cv_score = -numpy.inf

for combo in combos:
    score = column_scores(combo)
    cv_scores.append(score)

    if cv_scores[-1] > best_cv_score:
        best_cv_score = cv_scores[-1]
        best_combo = combo


print("Best combo: " + str(best_combo))
print("Which produced CV score: " + str(best_cv_score))

Pred_Train = Pred_Train[best_combo]

######################################################
# Logistic regression
C_pool = numpy.linspace(.1,1,40)
best_score = -numpy.inf
logistic_scores = []

for c in C_pool:
    LR1=LogisticRegression(C=c,max_iter=5000)
    logistic_scores.append(cross_val_score(LR1, Pred_Train, Targer_Train, cv=5).mean())

    if logistic_scores[-1]>best_score:
        best_score=logistic_scores[-1]
        best_c=c

LR1 = LogisticRegression(C=best_c, max_iter=5000)
LR1.fit(Pred_Train, Targer_Train)

Pred_test, Target_test = prep_penguins_data(test)
Pred_test = Pred_test[best_combo]

LR1.score(Pred_test, Target_test)

fig1, ax = plt.subplots(1)
ax.scatter(C_pool, logistic_scores)
ax.set(title="Best C parameter: " + str(best_c),
       xlabel="C parameter", ylabel="CV score")

##########################################################
# Support vector machine

gammas = numpy.linspace(0.1, 5, 100)
svm_scores = []
best_svm_score = -numpy.inf

for g in gammas:
    SVM = svm.SVC(gamma = g)
    svm_scores.append(cross_val_score(SVM, Pred_Train, Targer_Train, cv=5).mean())

    if svm_scores[-1] > best_svm_score:
        best_svm_score = svm_scores[-1]
        best_gamma = g

fig2, ax = plt.subplots(1)
ax.plot(gammas, svm_scores)
ax.set(title="Best SVM gamma: " + str(best_gamma) + " (" + str(best_svm_score*100) + "%)",
       xlabel="Gamma", ylabel="CV score")
plt.show()

SVM = svm.SVC(gamma=best_gamma)
SVM.fit(Pred_Train, Targer_Train)
SVM.score(Pred_Train, Targer_Train)
SVM.score(Pred_test, Target_test)