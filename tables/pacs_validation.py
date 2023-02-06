import pandas as pd 
import numpy as np 

metric = "pmi"

pmi = pd.read_csv(f"output/{metric}.csv")
pacs = pd.read_csv("inspire-harvest/database/pacs_codes.csv").set_index("code")

pmi.set_index('description', inplace=True)
pmi = pmi.transpose()
pmi = pmi.merge(pacs, how="inner", left_index=True, right_index=True)
pmi = pmi.transpose()
pmi.index.name = "topic"
pmi.reset_index(inplace=True)

pmi = (pd.melt(pmi, id_vars=["topic"], value_name='value', var_name="pacs"))
pmi = pmi[pmi["topic"] != "description"]

pmi = pmi.merge(pacs, how="inner", left_on="pacs", right_index=True)

df1 = pmi.sort_values(["topic", "value"], ascending=[True, False]).groupby("topic").head(10)
df1.set_index("topic", inplace=True)
df1.to_csv("output/validation.csv")

df1 = pmi.sort_values(["topic", "value"], ascending=[True, False]).groupby("topic").head(5)
df1.set_index("topic", inplace=True)

df2 = pmi.sort_values(["description", "value"], ascending=[True, False]).groupby("pacs").head(5)
df2.set_index("pacs", inplace=True)
df2.to_csv("output/validation_by_pacs.csv")

import textwrap

df1.reset_index(inplace = True)

# df1["description"] = df1["description"].apply(lambda s: "\\\\ ".join(textwrap.wrap(s, width=30)))
# df1["description"] = df1["description"].apply(lambda s: '\\begin{tabular}{@{}c@{}}' + s +'\\end{tabular}')

df1["description"] = df1["description"].str.replace("&", "\\&")

df1["topic"] = df1["topic"].apply(lambda s: "\\\\ ".join(textwrap.wrap(s, width=15)))
df1["topic"] = df1["topic"].apply(lambda s: '\\begin{tabular}{l}' + s +'\\end{tabular}')

df1["value"] = df1["value"].apply(lambda x: f"{x:.2f}")

df1.rename(columns = {
    "value": "pmi",
    "description": "PACS category"
}, inplace = True)

latex = df1.set_index(["topic", "PACS category"]).to_latex(
    columns=["pmi"],
    longtable=True,
    sparsify=True,
    multirow=True,
    multicolumn=True,
    position='H',
    column_format='p{0.25\\textwidth}|p{0.6\\textwidth}|p{0.15\\textwidth}',
    escape=False,
    caption="PACS categories most correlated to the topics derived with the unsupervised model. Correlation is measured as the mutual pointwise information (pmi).",
    label="table:full_topics_pacs_pmi"
)

with open("tables/topic_pacs_validation.tex", "w+") as fp:
    fp.write(latex)


# df2.reset_index(inplace = True)

# df2 = df2[df2["pacs"].isin(["11.30.Pb", "12.60.Jv", "14.80.Da", "14.80.Ly", "14.80.Nb", "04.65.+e"])]

# df2["topic"] = df2["topic"].str.replace("&", "\\&")

# df2["description"] = df2["description"].apply(lambda s: "\\\\ ".join(textwrap.wrap(s, width=15)))
# df2["description"] = df2["description"].apply(lambda s: '\\begin{tabular}{l}' + s +'\\end{tabular}')

# df2["value"] = df2["value"].apply(lambda x: f"{x:.2f}")

# df2.rename(columns = {
#     "value": "pmi",
#     "description": "Catégorie PACS"
# }, inplace = True)

# latex = df2.set_index(["Catégorie PACS", "topic"]).to_latex(
#     columns=["pmi"],
#     longtable=True,
#     sparsify=True,
#     multirow=True,
#     multicolumn=True,
#     position='H',
#     column_format='p{0.25\\textwidth}|p{0.6\\textwidth}|p{0.15\\textwidth}',
#     escape=False,
#     caption="Sujets les plus corrélés avec les catégories PACS supersymétriques. Le niveau de corrélation est estimé via l'information mutuelle ponctuelle (pmi).",
#     label="table:susy_pacs_pmi"
# ).replace("Continued on next page", "Suite page suivante")

# with open("analyses/susy_pacs_pmi.tex", "w+") as fp:
#     fp.write(latex)
