import pandas as pd 
import numpy as np

ans = []

def compile_survey(annotator, truth_file, answers_file):
    answers = pd.read_excel(answers_file)
    answers['question'] = answers['question'].fillna(method='ffill').astype(int)
    answers["answer"] = answers["1 topic or 2 topics ?"].astype(int)

    truth = pd.read_csv(truth_file)

    answers = answers.merge(truth, how="left", left_on="question", right_on="question")
    answers["correct"] = ((answers["answer"]==1)&answers["topic2"].isnull()) | ((answers["answer"]==2)&~answers["topic2"].isnull())
    return answers

# ans.append(compile_survey("lucas", "analyses/truth_lucas2.csv", "surveys/questions_lucas2_answered.xlsx"))
ans.append(compile_survey("acordeir", "analyses/truth_acordeir_weighted.csv", "surveys/acordeir.xlsx"))
ans.append(compile_survey("hessel", "analyses/truth_hessel_weighted.csv", "surveys/hessel.xlsx"))
ans = pd.concat(ans)

print(ans["correct"].mean())